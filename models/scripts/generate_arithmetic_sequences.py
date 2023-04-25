import h5py
import random
import os
import numpy as np
import copy

import tqdm

from contextlib import suppress
from itertools import chain, repeat

import torch

from models.scripts.data_model import Subject
from models.scripts.utils import create_db_session, to_secs
from random import choice

DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
OPERATORS = ['+', '-', '*', '/']
DECIMAL_NOTATION = ['.']
BRACKETS = ['(', ')']
EQUAL_SIGN = '='

PAD_IDX = -5
BOS_IDX = 2
EOS_IDX = 3

GRANULARITIES = ['glyph', 'stroke']
EXPR_MODES = ['all', 'digits', 'alphabets']
SAVE_MODES = ['unsolved', 'postfix', 'marked_postfix', 'solved']

DEF_AUGMENTATION = {'amount': 1,  # Amount to augment each glyph by
                    'combine': False,  # Determines if strategies should be combined
                    'strategies': [  # Define augmentation strategies along with their factors
                        ('scaleUp', 2),  # Scale glyph by 2%
                        ('scaleDown', 4),  # Reduce glyph by 4%
                        ('shiftX', 15),  # Shift along X axis by 15% (relative to client's viewport)
                        ('shiftY', 20),  # Shift along Y axis by 20% (relative to client's viewport)
                        ('squeezeX', 10),  # Squeeze along X axis by 10%
                        ('squeezeY', 10),  # Squeeze along Y axis by 10%
                    ]}


class SequenceGenerator:
    """
    Generates train, validation, and test set data

    param: vocab: vocabulary used to generate dataset
    param: allow_brackets: specify if complex expressions with brackets are allowed
    param: save_mode: specifies label to generate:
        ('unsolved': standard infix notation,
        'postfix': standard postfix notation,
        'marked_postfix': postfix but adds a separator mark at the end of the literals,
        'solved': saves results of the expressions)
    param: total_expressions: number of expressions to generate
    param: vector_size: number of touches to embed each stroke
    param: max_seq_len: Maximum length of an expression, default is 10
    param: padding_value (int): Value for padding the generated dataset to meet the required vector size (Def: -5)
    param: augmentation: dictionary of allowed glyph transformations to augment them
    param: train_split (float): Percentage of training set (defaults to 0.6, i.e. 60%)
    param: valid_split (float): Percentage of validation set (defaults to 0.2, i.e. 20%)
    param: scale_by_ar (bool): Flag to determine if y coordinate should be scaled by aspect ratio (Defaults to False)
    param: offset_delay (bool): Flag to determine if time delay between strokes should be considered (Defaults to False)
    param: granularity (str): Granularity of supervised dataset. (Def: 'glyph')
    param: end_of_glyph_value (int): Value to signify the end of a glyph in the generated dataset (Def: -1)
    param: end_of_stroke_value (int): Value to signify the end of a stroke in the generated dataset (Def: -2)
    param: include_time_feature (bool): Flag to indicate whether to include the time feature (Def: True)
    """

    CHUNK_SIZE = 256  # 8kb
    SAMPLING_RATE = 10e-3  # 5ms
    NUM_PROCESSES = 3  # One process per batch

    def __init__(self,
                 vocab,
                 allow_brackets=False,
                 save_mode="unsolved",
                 total_expressions=1000,
                 vector_size: int = 128,
                 db_path: str = "digit_schema.db",
                 max_seq_len: int = 12,
                 padding_value: int = -5,
                 augmentation=DEF_AUGMENTATION,
                 train_split: float = 0.6,
                 valid_split: float = 0.2,
                 use_subject=True,
                 scale_by_ar: bool = False,
                 sample_data: bool = False,
                 offset_delay: bool = False,
                 granularity: str = "stroke",
                 end_of_glyph_value: int = -4,
                 end_of_stroke_value: int = -3,
                 expr_mode: str = "digits",
                 include_time_feature: bool = False,
                 include_position_feature: bool = False):

        # Train and Validation split are within 0 and 1
        if augmentation is None:
            augmentation = {}
        assert (0 < train_split < 1), \
            "Train split should be between 0 and 1 (e.g 0.75)"
        assert (0 < valid_split < 1 - train_split), \
            f"Validation split should be between 0 and {1 - train_split}"

        # Ensure valid experiment mode and granularity are selected
        assert expr_mode in EXPR_MODES, f"Invalid experiment mode. Must be any of {EXPR_MODES}"
        assert granularity in GRANULARITIES, f"Invalid granularity. Must be any of {GRANULARITIES}"
        assert save_mode in SAVE_MODES, f"Invalid save_mode. Must be any of {SAVE_MODES}"

        self.vocab = vocab
        self.allow_brackets = allow_brackets
        self.save_mode = save_mode
        self.total_expressions = total_expressions
        self.expr_mode = expr_mode
        self.train_split = train_split
        self.vector_size = vector_size
        self.granularity = granularity
        self.augmentation = augmentation
        self.max_seq_len = max_seq_len
        self.validation_split = valid_split

        # bos/eos/pad tokens idx
        self.bos_idx = self._get_token_id_from_vocab('<bos>')
        self.eos_idx = self._get_token_id_from_vocab('<eos>')
        self.pad_idx = self._get_token_id_from_vocab('<pad>')

        # Misc
        self.db_path = db_path
        self._should_augment = any(augmentation)

        # Calculate test split
        self.test_split = 1 - (train_split + valid_split)

        # Boolean flags
        self.use_subject = use_subject
        self.scale_by_ar = scale_by_ar
        self.sample_data = sample_data
        self.offset_delay = offset_delay
        self.include_time_feature = include_time_feature
        self.include_position_feature = include_position_feature

        # If time is included, features = (x,y,t) else (x,y)
        self.n_features = 3 if self.include_time_feature else 2
        # If position is included, features = (x,y,t,p)
        self.n_features = self.n_features + 1 if self.include_position_feature else self.n_features

        # Constant tuples
        self.padding_value = padding_value
        self.end_of_glyph_value = end_of_glyph_value
        self.end_of_stroke_value = end_of_stroke_value

        # Generated dataset
        self._x_test = []
        self._y_test = []
        self._x_train = []
        self._y_train = []
        self._x_valid = []
        self._y_valid = []

        self.avg_glyph_strokes = 2  # Avg of 2 strokes per glyph from statistical analysis
        self.avg_glyph_touches = 128  # Avg of 128 touches per glyph from statistical analysis
        self.fname = f"expressions_{str(int(self.total_expressions / 1000)) + 'k'}_" \
                     f"{self.save_mode[0]}{'b' if self.allow_brackets else ''}"
        self.fpath = os.path.join("cache", "expressions_cache", f"{self.fname}.h5")

    def _get_token_id_from_vocab(self, token):
        """Get the token id from a vocabulary"""

        if type(self.vocab).__name__ == "ByteLevelBPETokenizer":
            token_id = self.vocab.token_to_id(token)

        else:  # `torchtext.vocab.Vocab`
            token_id = self.vocab.stoi[token]

        return token_id

    def _to_gen(self, x, y, mode, hf_file=None):
        """
        Converts (x, y) data to a generator and iteratively consumes
        it. This is to account for very large datasets that would be
        eventually consumed by the PyTorch DataLoader for batching
        """

        Y_count = 0
        X_count = [0, 0, 0]

        y = [i.decode("utf-8") for i in y]  # byte to str

        # Chunk size should be either the length of y, if possible, or the preset chunk size
        chunk_size = SequenceGenerator.CHUNK_SIZE if len(y) > SequenceGenerator.CHUNK_SIZE else len(y)

        # Range should have just one iteration if length of y is less than preset chunk size
        end_index = ((len(y) // chunk_size) + 1) if len(y) > SequenceGenerator.CHUNK_SIZE else 1

        for slice_index in range(0, end_index):
            next_slice_y = y[slice_index * chunk_size: (slice_index + 1) * chunk_size]
            next_slice_x = np.array(x[slice_index * chunk_size: (slice_index + 1) * chunk_size])

            Y_count += len(next_slice_y)
            X_count[2] = next_slice_x.shape[2]
            X_count[1] = next_slice_x.shape[1]
            X_count[0] += next_slice_x.shape[0]

            yield next_slice_x.tolist(), next_slice_y

        print(f"{mode.capitalize()} => {tuple(X_count), Y_count}")

        if hf_file:
            hf_file.close()  # Close the h5 file after

    def _pad(self, touches: list, size=None, padding=None):
        """Pad or chop off touches using the required size"""

        touches_length = len(touches)
        size = size or self.vector_size
        diff = abs(size - touches_length)

        # Vector size greater than touches?
        if size > touches_length:
            # Create padding list
            padding = padding or self.padding_value

            # Pad end of array up to vector size
            padded_touches = list(chain(touches, repeat(padding, diff)))

        # Touches greater than vector size?
        else:
            # Chop off difference to meet vector size
            padded_touches = list(chain(touches[0: (touches_length - diff)]))

        return padded_touches

    def _split(self, data_iter: list):
        """Splits a list of data_iter into the three sets"""

        # Get total length of data_iter
        data_iter_length = len(data_iter)

        train_index = int(self.train_split * data_iter_length)
        test_index = int(
            train_index + (self.validation_split * data_iter_length))

        # train set, validation set, test set
        t, v, ts = data_iter[0:train_index], data_iter[train_index: test_index], data_iter[test_index:]

        return t, v, ts

    def _generate_glyphs_from_expression(self, expression, subject_choices, sc):
        """
        Generates glyphs corresponding to a given `expression`
        If `self.use_subject` is True, random subjects are selected
        from the dataset and used as the source of the glyphs
        """

        # Hold possible subject and glyphs

        glyphs = []

        # For each character in the expression
        for char in expression:

            # Round-robin all subjects in the training set.
            # Ensures each subject is selected multiple times
            subject_to_use = None
            subj = subject_choices[sc]
            subject_glyphs = [gl for gs in subj.glyph_sequences for gl in gs.glyphs]

            glyph_choices = [i for i in subject_glyphs if i.ground_truth == char]
            if glyph_choices:
                subject_to_use = subj

            while not subject_to_use:

                sc += 1  # Increment index

                if sc >= len(subject_choices):
                    sc = 0  # Reset

                subj = subject_choices[sc]
                subject_glyphs = [gl for gs in subj.glyph_sequences for gl in gs.glyphs]

                glyph_choices = [i for i in subject_glyphs if i.ground_truth == char]

                if glyph_choices:
                    subject_to_use = subj

            glyph_choice = random.choice(glyph_choices)

            # Add to the glyphs list
            glyphs.append(glyph_choice)

        return glyphs, sc

    def _tensorize_string(self, trg_string):
        """Convert a string into a tensor"""

        if type(self.vocab).__name__ == "ByteLevelBPETokenizer":
            tsor = torch.tensor(self.vocab.encode(trg_string).ids)

        else:  # `torchtext.vocab.Vocab`
            tsor = torch.stack([torch.tensor(self.vocab.stoi[j])
                                for j in trg_string])

        return tsor

    def _granularize(self, glyphs, session):
        """Expand `glyph`s into required granularity"""

        expr_touches = []
        bos_vector = list(chain(repeat(self.bos_idx, self.vector_size)))
        eos_vector = list(chain(repeat(self.eos_idx, self.vector_size)))

        # Add beginning of sequence vector
        expr_touches.append(bos_vector)

        stroke_position = 0  # Track position of stroke in glyphs

        # For each glyph...
        for char in glyphs:

            stroke_start_time = 0  # Default start time
            if self.granularity == "glyph":
                touches = []

            # Use serialized version of glyph. It's faster
            char = char.serialize(session=session) if not isinstance(
                char, dict) else char

            # Delays between strokes
            stroke_delays = char['stroke_delays'].split(" ")

            # For each stroke in the glyph...
            for stroke_index, stroke in enumerate(char['strokes']):

                if self.granularity == "stroke":
                    touches = []

                # For each touch in the stroke...
                for index, touch in enumerate(stroke['touches'], start=1):

                    if self.granularity == "touch":
                        touches = []

                    x = touch['x']

                    # Scale y (or not) by aspect ratio
                    y = touch['y'] / \
                        char['ar'] if self.scale_by_ar else touch['y']

                    # Get timestamp in seconds
                    t = touch['timestamp'] * 1e-15 + stroke_start_time  # (s)

                    # Add to touches list
                    touches.append(x)  # x
                    touches.append(y)  # y
                    touches.append(
                        t) if self.include_time_feature else None  # t
                    touches.append(
                        stroke_position) if self.include_position_feature else None  # p

                    # Add touch to sequence dimension
                    if self.granularity == 'touch':
                        expr_touches.append(touches)

                        # Add end of stroke signal
                        if index == len(stroke['touches']):
                            expr_touches.append(
                                [self.end_of_stroke_value for _ in range(self.n_features)])

                # If time delay should be offset
                if self.offset_delay:
                    # Get delay before next stroke
                    delay_to_next_stroke = 0
                    with suppress(IndexError):
                        delay_to_next_stroke = stroke_delays[stroke_index]

                    # Rewrite the stroke's start time to the delay plus time
                    stroke_start_time = (t + to_secs(delay_to_next_stroke))

                if self.granularity == 'glyph':
                    # Add end of stroke signal
                    for _ in range(self.n_features):
                        touches.append(self.end_of_stroke_value)

                # Add end of glyph signal, if end of glyph
                # Should also offset stroke start time for
                # the next glyph by average glyph delays
                # if (stroke_index + 1) == len(char.strokes):
                #     touches.append(self.end_of_glyph_value)

                # If granularity is stroke, pad
                # Save entire [padded] stroke vector
                if self.granularity == 'stroke':
                    # Pad all touches for the stroke
                    # to the required `vector size`
                    touches = self._pad(touches)

                    # Save to expression ('Glyph')
                    expr_touches.append(touches)

            # if self.granularity == 'touch':
            # Add end of glyph signal
            # expr_touches.append([self.end_of_glyph_value for _ in range(self.n_features)])

            # Save entire [padded] glyph vector
            if self.granularity == 'glyph':
                # Pad all touches for the glyph
                # to the required `vector size`
                # touches is the concatenated touches
                # for all the strokes in the glyph
                touches = self._pad(touches)

                # Save to expression ('Glyph')
                expr_touches.append(touches)

            # Increment stroke posiiton used
            # when positon feature is enabled
            stroke_position += 1

        # Glyphs with multiple strokes should be accounted for
        if self.granularity == "glyph":
            max_size = self.max_seq_len

        elif self.granularity == "stroke":
            max_size = self.max_seq_len * self.avg_glyph_strokes

        elif self.granularity == "touch":
            max_size = self.max_seq_len * self.avg_glyph_touches

        # Padding tensor/list
        padding = list(chain(repeat(self.padding_value, self.vector_size)))

        # Add end of sequence vector
        expr_touches.append(eos_vector)

        # Pad dataset to the max_seq_len
        expr_touches = self._pad(expr_touches, size=max_size, padding=padding)

        return expr_touches  # Return the sequence dimension

    def _save_dataset(self, expanded_glyphs: list, exprs_batch: list, mode: str = 'train'):
        """Save related dataset"""

        if mode == 'test':
            x_dataset, y_dataset = self._x_test, self._y_test
        elif mode == 'valid':
            x_dataset, y_dataset = self._x_valid, self._y_valid
        elif mode == 'train':
            x_dataset, y_dataset = self._x_train, self._y_train
        else:
            raise

        # Save the whole expression
        for expr in exprs_batch:
            y_dataset.append(expr.encode('utf-8'))

        # Add to list of generated datasets
        for glyph in expanded_glyphs:
            x_dataset.append(glyph)

    def _augment_glyphs(self, glyphs: list, session):
        """
        Augment a list of glyphs
        Converts the passed glyphs to their json representation
        and adds them to the augmented list. Then, using those
        json-represented glyphs, augments glyphs up to the requested
        amount and using the selected strategies.

        The json-representation workaround is because creating new glyphs
        from their SQLAlchemy representation has proved troublesome
        """

        augmented_glyphs = []  # Storage for augmented glyphs

        # Loop the requested amount of augmented glyphs
        for glyph in glyphs:

            # Should strategies be combined?
            # combine_strategies = self.augmentation['combine']

            # Create new glyph from chosen glyph
            new_glyph = copy.deepcopy(glyph.serialize(session))

            # Get the augmentation strategy and associated factor
            strategy, factor = random.choice(self.augmentation['strategies'])

            # Force decimal
            factor = (factor % 100) / 100

            # For each stroke...
            for stroke in new_glyph['strokes']:

                # For each touch...
                for touch in stroke['touches']:
                    if strategy == 'shiftX':
                        touch['x'] = touch['x'] + factor

                    elif strategy == 'shiftY':
                        touch['y'] = touch['y'] + factor

                    elif strategy == 'squeezeX':
                        aug_m = [[1 + factor, 0], [0, 1]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'squeezeY':
                        aug_m = [[1, 0], [0, 1 + factor]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'scaleUp':
                        aug_m = [[1 + factor, 0], [0, 1 + factor]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'scaleDown':
                        aug_m = [[1 - factor, 0], [0, 1 - factor]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    elif strategy == 'skew':
                        deg = factor * np.pi / 180  # To radians
                        aug_m = [[np.cos(deg), -(np.sin(deg))],
                                 [np.sin(deg), np.cos(deg)]]
                        x, y = np.array((touch['x'], touch['y'])).dot(
                            aug_m).tolist()
                        touch['x'], touch['y'] = x, y

                    # More
                    else:
                        pass

            # Save augmented glyph (spatial)
            augmented_glyphs.append(new_glyph)

        return augmented_glyphs

    def _load_dataset_from_cache(self, cache_file=None, mode='train'):
        """Loads dataset from a cache file"""

        if not cache_file:
            cache_file = self.fpath

        if mode == "train":
            X = "X_train"
            Y = "Y_train"

        elif mode == "valid":
            X = "X_valid"
            Y = "Y_valid"

        elif mode == "test":
            X = "X_test"
            Y = "Y_test"

        else:
            raise AttributeError(f"{mode} is an invalid mode")

        hf = h5py.File(cache_file, 'r')

        # Return generator
        return self._to_gen(hf[X], hf[Y], mode, hf)

    def get_all_subjects(self, session, mode=None):
        """
        Get all subjects in dataset
        If `mode` is passed, then the corresponding subjects
        for that mode (e.g. `test`, or `train`) is returned.
        """

        subjects_ = []

        if self.expr_mode == "digits":
            expr = [1, 4]
        elif self.expr_mode == "alphabets":
            expr = [1, 3]  # Note that some subjects have missing characters.
        else:  # Implicit "all"
            expr = [2]

        # Ascending order
        for subject in session.query(Subject):
            for gs in subject.glyph_sequences:
                if gs.experiment in expr and subject not in subjects_:
                    subjects_.append(subject)

        # Seeded shuffle
        random.seed(5050)
        random.shuffle(subjects_)

        # Split the [available] subjects into train, validation, and test subjects
        subjects = self._split(subjects_)

        # Ensure subjects are unique across each split mode
        assert len(set(subjects[0]) & set(subjects[2])) == 0
        assert len(set(subjects[0]) & set(subjects[1])) == 0
        assert len(set(subjects[1]) & set(subjects[2])) == 0

        if mode:
            if mode == 'test':
                index = 2
            elif mode == 'valid':
                index = 1
            else:  # Implicit train
                index = 0

            return subjects[index]

        print("Subjects are unique!\n")
        return subjects

    def cache_generated_dataset(self, fname=None):
        """Save generated  data to disk as a h5 file"""

        print("\nCaching...")

        if len(self._x_test) == 0:
            return "Caching failed. No generated data."

        if not fname:
            fname = self.fname

        fpath = os.path.join("cache", "expressions_cache", f"{fname}.h5")

        # Output is a list of expression
        y_test = np.array(self._y_test, dtype='S')
        y_valid = np.array(self._y_valid, dtype='S')
        y_train = np.array(self._y_train, dtype='S')

        with h5py.File(fpath, 'w') as hf:
            hf.create_dataset("Y_test", compression="gzip",
                              chunks=True, data=y_test)
            hf.create_dataset("Y_valid", compression="gzip",
                              chunks=True, data=y_valid)
            hf.create_dataset("Y_train", compression="gzip",
                              chunks=True, data=y_train)
            hf.create_dataset("X_test", compression="gzip",
                              chunks=True, data=np.array(self._x_test))
            hf.create_dataset("X_valid", compression="gzip",
                              chunks=True, data=np.array(self._x_valid))
            hf.create_dataset("X_train", compression="gzip",
                              chunks=True, data=np.array(self._x_train))

        print(f"Dataset saved to {fpath}.")

    def generate_src_from_trg_string(self, trg_string, subject, session, seed=None):
        """Generate source and target tensors from an input string"""

        g = []
        subj_glyphs = [j for i in subject.glyph_sequences for j in i.glyphs]

        if seed:
            random.seed(seed)
        for c in trg_string:
            if c == " ":
                continue  # Skip blanks
            try:
                ch = choice([i for i in subj_glyphs if i.ground_truth == c])
            except:
                print(c)
                print(sorted(list(set([i.ground_truth for i in subj_glyphs]))))
                break

            g.append(ch)

        # Expand the glyphs into strokes or glyphs
        src = torch.tensor(self._granularize(g, session))

        # Convert each char in the expr to a tensor
        y = self._tensorize_string(trg_string)

        # Add bos/eos and pad up tokens up to max_seq_len
        diff = (self.max_seq_len - (y.shape[0])) - 2
        y_ = torch.cat([torch.tensor([self.bos_idx]),
                        y, torch.tensor([self.eos_idx]),
                        torch.tensor(list(repeat(torch.tensor([self.pad_idx]), diff)))], dim=0)

        # Because equally-padded tensors have float
        trg = torch.tensor(y_, dtype=torch.int64)

        # Return src and trg
        return src.unsqueeze(0), trg.unsqueeze(0)

    def _create_dataset(self):
        """Create an empty dataset"""

        self._x_test = []
        self._y_test = []
        self._x_train = []
        self._y_train = []
        self._x_valid = []
        self._y_valid = []

    def _init_cache(self, fname=None):
        """Initialize cache"""

        print("\nInitializing cache...")

        if not fname:
            fname = self.fpath

        if os.path.exists(fname):
            os.remove(fname)
        self.dtype = np.dtype('S15')

        with h5py.File(fname, 'w') as hf:
            ms = (self.avg_glyph_strokes * self.max_seq_len, self.vector_size)

            # Create the actual datasets
            hf.create_dataset('X_test', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('X_train', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('X_valid', compression="gzip",
                              chunks=True, maxshape=(None, *ms), shape=(0, *ms))
            hf.create_dataset('Y_test', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)
            hf.create_dataset('Y_train', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)
            hf.create_dataset('Y_valid', compression="gzip",
                              chunks=True, maxshape=(None,), shape=(0,), dtype=self.dtype)

    def _update_cache(self):
        """
        Save generated dataset to filesystem as a .h5 file
        """
        print("Updating cache")

        y_test = np.array(self._y_test, dtype=self.dtype)
        y_valid = np.array(self._y_valid, dtype=self.dtype)
        y_train = np.array(self._y_train, dtype=self.dtype)

        x_test = np.array(self._x_test)
        x_train = np.array(self._x_train)
        x_valid = np.array(self._x_valid)

        # Add updates to h5 cache
        with h5py.File(self.fpath, 'a') as hf:
            if self._y_test:
                hf["Y_test"].resize((hf["Y_test"].shape[0] + y_test.shape[0]), axis=0)
                hf["X_test"].resize(hf["X_test"].shape[0] + x_test.shape[0], axis=0)
                hf["Y_test"][-y_test.shape[0]:] = y_test.astype(self.dtype)
                hf["X_test"][-x_test.shape[0]:] = x_test
            if self._y_train:
                hf["Y_train"].resize((hf["Y_train"].shape[0] + y_train.shape[0]), axis=0)
                hf["X_train"].resize((hf["X_train"].shape[0] + x_train.shape[0]), axis=0)
                hf["Y_train"][-y_train.shape[0]:] = y_train.astype(self.dtype)
                hf["X_train"][-x_train.shape[0]:] = x_train
            if self._y_valid:
                hf["Y_valid"].resize((hf["Y_valid"].shape[0] + y_valid.shape[0]), axis=0)
                hf["X_valid"].resize((hf["X_valid"].shape[0] + x_valid.shape[0]), axis=0)
                hf["Y_valid"][-y_valid.shape[0]:] = y_valid.astype(self.dtype)
                hf["X_valid"][-x_valid.shape[0]:] = x_valid

        # Reset for next batch
        self._create_dataset()

    def generate_from_cache(self, cache_file=None, mode=None):
        """Generate the train, validaton, and test datasets from a cache file"""

        if not cache_file:
            cache_file = self.fpath

        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"{cache_file} does not exist.")

        print(f"Using cached dataset file in {cache_file}")

        if mode:
            if mode == 'test':
                return self._load_dataset_from_cache(cache_file, "test")
            elif mode == 'valid':
                return self._load_dataset_from_cache(cache_file, "valid")
            elif mode == 'train':
                return self._load_dataset_from_cache(cache_file, "train")
            else:
                raise

        else:
            test = self._load_dataset_from_cache(cache_file, "test")
            valid = self._load_dataset_from_cache(cache_file, "valid")
            train = self._load_dataset_from_cache(cache_file, "train")

        return train, valid, test

    def _random_decimal(self, number):
        number = round(float(number))
        t = random.random()
        if t < .5:
            return number
        decimal_part = round(random.random(), 1)
        return number + decimal_part

    def _generate_random_expression(self):
        MAX_NUM = 10
        if not self.allow_brackets:
            max_num_operands = random.randint(1, int(self.max_seq_len / 2 - 1))
            expr = ''
            total_len = 0

            for i in range(max_num_operands):
                random_number = str(self._random_decimal(random.random() * MAX_NUM))
                if total_len + len(random_number) > self.max_seq_len - 1:
                    expr += random_number[0]
                    break
                total_len += len(random_number)
                expr += random_number

                if total_len >= self.max_seq_len - 2:
                    break
                t = random.random()
                if t < .25 and i > 1:
                    break
                expr += random.choice(OPERATORS)
                total_len += 1

            else:
                random_number = str(self._random_decimal(random.random() * MAX_NUM))
                if total_len + len(random_number) > self.max_seq_len - 1:
                    random_number = random_number[0]
                expr += random_number
            return expr

        else:  # allow brackets is true
            final_expr = random.choice(['E' + op + 'E' for op in OPERATORS])
            allowed_expressions = DIGITS + ['(E' + op + 'E)' for op in OPERATORS] + ['E' + op + 'E' for op in OPERATORS]
            total_len = 3
            next_index = final_expr.find('E')
            while next_index >= 0:
                new_len = self.max_seq_len
                while total_len + new_len > self.max_seq_len:
                    new_expr = random.choice(allowed_expressions)
                    if new_expr in DIGITS:
                        new_expr = str(self._random_decimal(new_expr))
                    new_len = len(new_expr)
                final_expr = final_expr.replace('E', new_expr, 1)
                total_len += new_len - 1
                next_index = final_expr.find('E')
            return final_expr

    def _infix_to_postfix(self, expression):
        Priority = {'+': 1, '-': 1, '*': 2, '/': 2}  # dictionary having priorities of Operators
        stack = []  # initialization of empty stack
        output = ''

        for character in expression:
            if character not in OPERATORS + BRACKETS:  # if an operand append in postfix expression
                output += character

            elif character == '(':  # else Operators push onto stack
                stack.append('(')

            elif character == ')':
                while stack and stack[-1] != '(':
                    output += stack.pop()
                stack.pop()

            else:
                while stack and stack[-1] != '(' and Priority[character] <= Priority[stack[-1]]:
                    output += stack.pop()
                stack.append(character)
        while stack:
            output += stack.pop()
        return output

    def _infix_to_marked_postfix(self, expression):
        Priority = {'+': 1, '-': 1, '*': 2, '/': 2}  # dictionary having priorities of Operators
        stack = []  # initialization of empty stack
        output = ''

        new_expression = ''
        for character1, character2 in zip(expression, expression[1:]):
            new_expression = new_expression + character1
            if character1 in DIGITS and character2 in OPERATORS + BRACKETS:
                new_expression = new_expression + ','
        new_expression = new_expression + expression[-1]
        if expression[-1] in DIGITS:
            new_expression = new_expression + ','

        for character in new_expression:
            if character not in OPERATORS + BRACKETS:  # if an operand append in postfix expression
                output += character

            elif character == '(':  # else Operators push onto stack
                stack.append('(')

            elif character == ')':
                while stack and stack[-1] != '(':
                    output += stack.pop()
                stack.pop()

            else:
                while stack and stack[-1] != '(' and Priority[character] <= Priority[stack[-1]]:
                    output += stack.pop()
                stack.append(character)
        while stack:
            output += stack.pop()
        return output

    def generate(self):
        """
        Generate the train, validation, and test datasets.
        """

        train = []
        valid = []
        test = []
        self._init_cache(None)

        generated_expressions = []
        for _ in tqdm.tqdm(range(self.total_expressions)):
            expression = self._generate_random_expression() + EQUAL_SIGN
            generated_expressions.append(expression)

        # Split expressions into train, validation, test
        train_exprs, valid_exprs, test_exprs = self._split(generated_expressions)

        print("Generating datasets...\n")
        for (batch, mode) in [(test_exprs, 'test'), (train_exprs, 'train'), (valid_exprs, 'valid')]:

            session = create_db_session(self.db_path)
            subjects = self.get_all_subjects(session, mode)

            sc = 0  # Subject counter index. Used to recycle subjects
            et, etw = [], []  # Storage for generated touches and corresponding ground truth

            for index, expr in enumerate(tqdm.tqdm(batch, desc=f"{mode.capitalize()} set progress")):
                # 'boy' -> 'Glyph (b)', 'Glyph (o)', 'Glyph (y)'
                glyphs, sc = self._generate_glyphs_from_expression(expr, subjects, sc)

                # Expand the glyphs into strokes or glyphs
                expr_touches = self._granularize(glyphs, session)

                # Save expanded touches
                et.append(expr_touches)

                # Save appropriate expression
                if self.save_mode == 'solved':
                    try:
                        label = eval(expr[:-1])
                        label = round(label, 2)
                        label = str(label)
                    except:
                        label = "IMP"
                elif self.save_mode == 'postfix':
                    label = self._infix_to_postfix(expr[:-1]) + EQUAL_SIGN
                elif self.save_mode == 'marked_postfix':
                    label = self._infix_to_marked_postfix(expr[:-1]) + EQUAL_SIGN
                else:
                    label = expr
                label = label.strip("'").strip('"').strip('[').strip(']')
                etw.append(label)

                if self._should_augment:  # If augmenting...
                    for _ in range(self.augmentation['amount']):
                        aug_glyphs = self._augment_glyphs(glyphs, session)

                        # Expand the glyphs depending on granularity
                        aug_expr_touches = self._granularize(
                            aug_glyphs, session)

                        # Save expanded touches
                        et.append(aug_expr_touches)
                        if self.save_mode == 'solved':
                            try:
                                label = eval(expr[:-1])
                                label = round(label, 2)
                                label = str(label)
                            except:
                                label = "IMP"
                        elif self.save_mode == 'postfix':
                            label = self._infix_to_postfix(expr[:-1]) + EQUAL_SIGN
                        elif self.save_mode == 'marked_postfix':
                            label = self._infix_to_marked_postfix(expr[:-1]) + EQUAL_SIGN
                        else:
                            label = expr
                        label = label.strip("'").strip('"').strip('[').strip(']')
                        etw.append(label)

            assert len(et) == len(etw)
            print(f"Processed {mode} batch... (Total={len(etw)}).")

            # Save to class
            self._save_dataset(et, etw, mode=mode)
            # Return generators to conserve memory
            if self._y_test:
                test = self._to_gen(self._x_test, self._y_test, "test")
            if self._y_train:
                train = self._to_gen(self._x_train, self._y_train, "train")
            if self._y_valid:
                valid = self._to_gen(self._x_valid, self._y_valid, "valid")

            self._update_cache()

        return train, valid, test

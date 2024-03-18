from enum import Enum
from collections import defaultdict
import copy

class MotionClass(Enum):
    """Represents the different motion classes with values corresponding to their index in alphabetical order."""
    Centering = 0
    MakingHole = 1
    Pressing = 2
    Raising = 3
    Smoothing = 4
    Sponge = 5
    Tightening = 6

    @classmethod
    def names(cls):
        """Returns all the class names in a list in alphabetical order."""
        return [m.name for m in MotionClass]

    @classmethod
    def convert(cls, input):
        """
        Converts the input argument to a MotionClass object. Used by the different classes for indexing.
        
        Args:
            input: A string, integer or MotionClass object
        
        Returns:
            A MotionClass object.
        
        Raises:
            Exception: If the the provided input cannot be interpreted as a MotionClass.
        """
        if type(input) is MotionClass:
            return input
        elif type(input) is str:
            return MotionClass[input]
        elif type(input) is int:
            return MotionClass(input)
        else:
            raise Exception("Cannot convert type '{}' to MotionClass.".format(type(input)))


class Marker(Enum):
    """Represents the different Vicon markers with values corresponding to their order of appearance in the structure of the dataset."""
    LIWR = 0
    LOWR = 1
    LIHAND = 2
    LOHAND = 3
    LTHM3 = 4
    LTHM6 = 5
    LIDX3 = 6
    LIDX6 = 7
    LMID0 = 8
    LMID6 = 9
    LRNG3 = 10
    LRNG6 = 11
    LPNK3 = 12
    LPNK6 = 13
    RIWR = 14
    ROWR = 15
    RIHAND = 16
    ROHAND = 17
    RTHM3 = 18
    RTHM6 = 19
    RIDX3 = 20
    RIDX6 = 21
    RMID0 = 22
    RMID6 = 23
    RRNG3 = 24
    RRNG6 = 25
    RPNK3 = 26
    RPNK6 = 27

    @classmethod
    def connections(cls, index_only=False):
        """Returns the arbitrary connections between the markers for use as edges during data visualization."""
        left_connections = [(Marker(0), Marker(1)), (Marker(0), Marker(4)), (Marker(4), Marker(5)), (Marker(0), Marker(2)), (Marker(2), Marker(6)), (Marker(6), Marker(7)), (Marker(2), Marker(8)), (Marker(8), Marker(9)), (Marker(8), Marker(3)), (Marker(2), Marker(3)), (Marker(3), Marker(10)), (Marker(10), Marker(11)), (Marker(3), Marker(12)), (Marker(12), Marker(13)), (Marker(3), Marker(1))]
        right_connections = [(Marker(l.value + 14), Marker(r.value + 14)) for (l, r) in left_connections]

        connections = left_connections + right_connections
        
        if index_only:
            return [(l.value, r.value) for (l, r) in connections]
        else:
            return connections

    @classmethod
    def convert(cls, input):
        """
        Converts the input argument to a Marker object. Used by the different classes for indexing.
        
        Args:
            input: A string, integer or Marker object
        
        Returns:
            A Marker object.
        
        Raises:
            Exception: If the the provided input cannot be interpreted as a Marker.
        """
        if type(input) is Marker:
            return input
        elif type(input) is str:
            return Marker[input]
        elif type(input) is int:
            return Marker(input)
        else:
            raise Exception("Cannot convert type '{}' to Marker.".format(type(input)))
            

class Data:
    """Represents all the data in a dataset as a collection of MotionCapture objects."""
    def __init__(self, data=None):
        if not data:
            data = defaultdict(list)
        self.data = data

    def append(self, mocap):
        """Adds a new mocap to the collection."""
        self.data[mocap.motionclass].append(mocap)

    def __getitem__(self, key):
        key = MotionClass.convert(key)
        return self.data[key]

    def __len__(self):
        return sum([len(l) for l in self.data.values()])

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        length = len(self.data)
        res = "{"
        for idx, d in enumerate(self):
            res += " " if not idx == 0 else ""
            res += "{} ({} examples)".format(d.name, len(self[d]))
            res += ",\n" if idx != length - 1 else "}\n"
        return res
        
    def filter_markers(self, markers, keep=False):
        """Drops (or keeps) only specific marker data from the entire dataset."""
        new_data = Data(copy.deepcopy(self.data))
        for mocap_list in new_data.data.values():
            for mocap in mocap_list:
                mocap.filter_markers(markers, keep=keep)
        return new_data


class MotionCapture:
    """Represents a sequence of DataPoints."""
    def __init__(self, class_name, sequence):
        self.sequence = sequence # list of data points
        self.motionclass = MotionClass[class_name] if class_name else None

    def __getitem__(self, key):
        return self.sequence[key]

    def __str__(self):
        return "\n".join([str(x) for x in self.sequence])

    def __len__(self):
        return len(self.sequence)

    def filter_markers(self, markers, keep=False):
        """Drops (or keeps) only specific marker data from each datapoint in this mocap."""
        for i in range(len(self.sequence)):
            self.sequence[i].filter(markers, keep=keep)
                             

class DataPoint:
    """Represents a single frame of a motion capture."""
    def __init__(self, frame_num, marker_positions):
        self.frame_num = frame_num
        self.marker_data = {}
        for i in range(len(marker_positions)):
            self.marker_data[Marker(i)] = marker_positions[i]

    def __getitem__(self, key):
        if type(key) in (Marker, str, int):
            return self.marker_data[Marker.convert(key)]

        # recursive calls
        elif type(key) in (tuple, list):
            return [self[key] for k in key]
        elif type(key) is slice:
            return [self[i] for i in range(key.start if key.start else 0, key.stop if key.stop else len(self.Marker_data), key.step if key.step else 1)]
        
        else:
            raise Exception("Attempted to index DataPoint object with invalid key.")

    def __str__(self):
        return "{}: {}".format(self.frame_num, self.marker_data)

    def __iter__(self):
        return iter(self.marker_data.values())

    def values(self, transpose=False):
        """Returns the flattened marker position data as a list of floats."""
        v = sum([list(tup) for tup in list(self)], start=[])
        if transpose:
            v = [[x] for x in v]
        return v

    def filter(self, markers, keep=False):
        """Drops (or keeps) only specific marker data from this datapoint."""
        markers = [Marker[m] if type(m) is str else Marker(m) for m in markers]
        if keep:
            markers = [m for m in list(Marker) if m not in markers]
        for m in markers:
            self.marker_data.pop(m, None)

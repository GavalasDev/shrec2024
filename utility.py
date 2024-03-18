import os

from data import Marker, DataPoint, MotionCapture, MotionClass, Data

class DataParser:
    """Allows for parsing of an entire dataset under the provided path."""
    def __init__(self, path):
        self.path = path

    def parse(self, validation_percentage=0, verbose=False):
        """
        Parses all data under the provided path.
        
        Args:
            validation_percentage: A float representing the percentage of data to be separated as validation data (e.g. 0.2).
        
        Returns:
            A single Data object if validation_percentage=0 or a pair of Data objects in the form of (training, validation).
        """
        train = Data()
        valid = Data()
        for class_name in [c.name for c in MotionClass]:
            directory = os.path.join(self.path, class_name)
            files = sorted(os.listdir(directory))
            train_len = round(len(files) * (1-validation_percentage))
            for idx, file in enumerate(files):
                file_path = os.path.join(directory, file)
                if verbose:
                    print("reading file {}".format(file_path))
                mocap = self.parse_single(file_path, class_name)
                if idx < train_len:
                    train.append(mocap)
                else:
                    valid.append(mocap)
                
        return train if validation_percentage==0 else (train, valid)


    @classmethod
    def parse_single(cls, filepath, class_name=None):
        """
        Parses a single input sequence.

        Args:
            filepath: The path of the input sequence file.

        Returns:
            A motioncapture object.
        """
        with open(filepath) as f:
            sequence = [] # list of datapoints
            for line in f:
                line = line.split(';')
                marker_positions = [] # list of (x,y,z) tuples
                frame_num = line[0]
                for i in range(1, len(line)-1, 3):
                    marker_positions.append(tuple([float(x) for x in line[i:i+3]]))
                data_point = DataPoint(frame_num, marker_positions)
                sequence.append(data_point)
        mocap = MotionCapture(class_name, sequence)
        return mocap
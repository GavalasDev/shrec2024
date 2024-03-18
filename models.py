import numpy as np
from hmmlearn import hmm

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from data import Data, MotionClass, MotionCapture

class GMMHMMPatched(hmm.GMMHMM):
    # monkey patch to avoid errors
    def _do_mstep(self, stats):
        super()._do_mstep(stats)
        np.nan_to_num(self.covars_, copy=False)
        np.nan_to_num(self.weights_, copy=False)
        np.nan_to_num(self.means_, copy=False)

class PredictionResults:
    """Tracks the current validation or testing results and allows for easily visualizing them."""
    def __init__(self, class_num=7, confusion_array=None, accuracy=0):
        self.confusion_array = confusion_array if confusion_array else np.zeros((class_num, class_num))
        self.accuracy = accuracy
        self._correct = 0
        self._total = 0
        
    def append(self, prediction, correct):
        """
        Add a new prediction attempt to the results.
        
        Args:
            prediction: An object that can be interpreted as a MotionClass (i.e. str, int, or MotionClass) representing a models prediction.
            correct: An object that can be interpreted as a MotionClass (i.e. str, int, or MotionClass) representing the correct class.
        """
        prediction_class_idx = MotionClass.convert(prediction).value
        correct_class_idx = MotionClass.convert(correct).value

        self.confusion_array[correct_class_idx, prediction_class_idx] += 1
        self._total += 1
        if correct_class_idx == prediction_class_idx:
            self._correct += 1
        self.accuracy = self._correct / self._total

    def __str__(self):
        return "Confusion Matrix:\n{}\nOverall Accuracy: {}%".format(self.confusion_array, self.accuracy*100)

    def show(self, label_scale=1, font_size=12, figsize=(4, 3), cmap="PuBu"):
        """Plots a confusion matrix and prints the overall accuracy."""
        df_cm = pd.DataFrame(self.confusion_array, MotionClass.names(), MotionClass.names())
        plt.figure(figsize=figsize)
        sn.set(font_scale=label_scale) # for label size
        sn.heatmap(df_cm, annot=True, cmap=cmap, annot_kws={"size": font_size})
        plt.show()
        print("Overall Accuracy: {}% ({}/{})".format(self.accuracy*100, self._correct, self._total))
        
    

class HMModel:
    """Architecture based on [@papadopoulosRealTimeSkeletonTrackingBasedHuman2014](https://link.springer.com/chapter/10.1007/978-3-319-04114-8_40)) with a GMM for each motion class."""
    def __init__(self, classes=list(MotionClass), n_components=3, n_mix=3, n_iter=100, tol=0.01, seed=39, h_type='position', downsample_step=1, verbose=False):
        self.n_components = n_components
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.tol = tol
        self.h_type = h_type
        self.verbose = verbose
        self.downsample = downsample_step

        np.random.seed(seed)
        
        self.hmms = {}
        # one HMM for each class 
        for c in classes:
            self.hmms[c] = self.init_gmm()

    def init_gmm(self):
        m = GMMHMMPatched(n_components=self.n_components, n_mix=self.n_mix, implementation='scaling', covariance_type="full", tol=self.tol, n_iter=self.n_iter, verbose=self.verbose)
        return m

    def reset_model(self, motionclass):
        motionclass = MotionClass.convert(motionclass)
        self.hmms[motionclass] = self.init_gmm()
        return(self.hmms[motionclass])

    
    def compute_h(self, mocap):
        """
        Computes the h vector of a MotionCapture.
        
        h represents the string of data fed to the GMMs. The type of encoding is chosen during initialization using the h_type parameter.
        Available h_types:
        - position: list of (x, y, z) of markers
        - velocity: list of (Δx, Δy, Δz) of markers
        - velpos: list of interlaced position, velocity values i.e. (x, y, z, Δx, Δy, Δz) for each marker
        - grip: list of velocity of hand (based on IHAND marker) + distance of each fingertip from IHAND
        - veldist: list of interlaced distance from IHAND, velocity for each fingertip
        """
        if self.h_type == 'position':
            h = sum([dp.values(transpose=True) for dp in mocap[::self.downsample]], start=[]) # concat all coordinate values
        elif self.h_type == 'velocity':
            h = []
            for i in range(1,len(mocap),self.downsample): # drops first frame for velocity calculation
                v = (np.array(list(mocap[i])) - np.array(list(mocap[i-1]))).reshape(-1,1).tolist()
                h += v
        elif self.h_type == 'velpos':
            h = []
            for i in range(1,len(mocap),self.downsample): # drops first frame for velocity calculation
                sens = np.array(list(mocap[i])).reshape(-1, 1)
                sens_prev = np.array(list(mocap[i-1])).reshape(-1, 1)
                v = (sens - sens_prev).tolist()
                h += v
                p = sens.tolist()
                h += p
        elif self.h_type == 'grip':
            h = []
            for i in range(1,len(mocap),self.downsample): # drops first frame for velocity calculation
                for H in ('L', 'R'):
                    ihand = np.array(list(mocap[i][f'{H}IHAND'])).reshape(-1,1)
                    ihand_prev = np.array(list(mocap[i-1][f'{H}IHAND'])).reshape(-1,1)
                    # x,y,z velocity of IHAND marker
                    v_palm = (ihand - ihand_prev).tolist() 
                    h += v_palm
                    # distance of each fingertip from IHAND
                    for S in ['THM6', 'IDX6', 'MID6', 'RNG6', 'PNK6']:
                        sens = np.array(list(mocap[i][f'{H}{S}'])).reshape(-1,1)
                        d = [[np.linalg.norm(sens - ihand)]]
                        h += d
        elif self.h_type == 'veldist':
            h = []
            for i in range(1,len(mocap),self.downsample): # drops first frame for velocity calculation
                for H in ('L', 'R'):
                    ihand = np.array(list(mocap[i][f'{H}IHAND'])).reshape(-1,1)
                    for M in ['THM6', 'IDX6', 'MID6', 'RNG6', 'PNK6']:
                        mark = np.array(list(mocap[i][f'{H}{M}'])).reshape(-1,1)
                        mark_prev = np.array(list(mocap[i-1][f'{H}{M}'])).reshape(-1,1)
                        v = (mark - mark_prev).tolist()
                        h += v
                        d = [[np.linalg.norm(mark - ihand)]]
                        h += d
        elif self.h_type == 'angle':
            h = []
            for i in range(1,len(mocap),self.downsample):
                for H in ('L', 'R'):
                    iwr =  np.array(list(mocap[i][f'{H}IWR']))
                    ihand = np.array(list(mocap[i][f'{H}IHAND']))
                    ohand = np.array(list(mocap[i][f'{H}OHAND']))

                    # calculate orthogonal vectors
                    n1 = (ihand - ohand) / np.linalg.norm(ihand - ohand)
                    u = (ihand - iwr) / np.linalg.norm(ihand - iwr)
                    n3 = np.cross(n1, u) / np.linalg.norm(np.cross(n1, u))
                    n2 = np.cross(n3, n1)

                    for M in ['THM6', 'IDX6', 'MID6', 'RNG6', 'PNK6']:
                        mark = np.array(list(mocap[i][f'{H}{M}']))
                        phi = np.arccos(np.dot(mark-iwr, n2)/(np.linalg.norm(mark - iwr) * np.linalg.norm(n2)))
                        h.append([phi])
                        theta = np.arctan(np.dot(mark-iwr, n1)/np.dot(mark-iwr, n3))
                        h.append([theta])
        elif self.h_type == 'angvel':
            h = []
            for i in range(0,len(mocap),self.downsample):
                for H in ('L', 'R'):
                    iwr =  np.array(list(mocap[i][f'{H}IWR']))
                    ihand = np.array(list(mocap[i][f'{H}IHAND']))
                    ohand = np.array(list(mocap[i][f'{H}OHAND']))

                    iwr_prev =  np.array(list(mocap[i-1][f'{H}IWR']))
                    v_iwr = iwr - iwr_prev

                    h += v_iwr.reshape(-1, 1).tolist()

                    # calculate orthogonal vectors
                    n1 = (ihand - ohand) / np.linalg.norm(ihand - ohand)
                    u = (ihand - iwr) / np.linalg.norm(ihand - iwr)
                    n3 = np.cross(n1, u) / np.linalg.norm(np.cross(n1, u))
                    n2 = np.cross(n3, n1)

                    for M in ['THM6', 'IDX6', 'MID6', 'RNG6', 'PNK6']:
                        mark = np.array(list(mocap[i][f'{H}{M}']))
                        mark_prev = np.array(list(mocap[i-1][f'{H}{M}']))

                        v_mark = mark - mark_prev
                        
                        phi = np.arccos(np.dot(mark-iwr, n2)/(np.linalg.norm(mark - iwr) * np.linalg.norm(n2)))
                        h.append([phi])
                        theta = np.arctan(np.dot(mark-iwr, n1)/np.dot(mark-iwr, n3))
                        h.append([theta])

                        phi_vel = np.arccos(np.dot(v_mark-v_iwr, n2)/(np.linalg.norm(v_mark - v_iwr) * np.linalg.norm(n2)))
                        h.append([phi_vel])

                        theta_vel = np.arctan(np.dot(v_mark-v_iwr, n1)/np.dot(v_mark-v_iwr, n3))
                        h.append([theta_vel])
        return h

    def _train_generator(self, data: Data, cross_validate=False):
        results = PredictionResults() if cross_validate else None
        if cross_validate:    
            leave_out_list = [] # [(MotionClass['Centering'], 0), (MotionClass['Centering'], 1), ..., (MotionClass['Centering'], 12), (MotionClass['MakingHole'], 0), ...]
            for d in list(data):
                for i in range(len(data[d])):
                    leave_out_list.append((d, i))
        else:
            leave_out_list = []

        previous_left_out_class = -1

        for i in range(len(leave_out_list)+1):
            
            if i == len(leave_out_list):
                # final (full) pass
                cross_validate = False
                left_out = (None, None)
            else:
                left_out = leave_out_list[i]
            
            if self.verbose:
                print("-" * 150)
                if not cross_validate:
                    print("Performing full training")
                else:
                    print("Leaving out sample #{} of class {}".format(left_out[1], left_out[0].name))
            
            # fit the model of every class
            for motionclass in self.hmms:

                if not (previous_left_out_class == -1 or motionclass == left_out[0] or motionclass == previous_left_out_class):
                    if self.verbose:
                        print("HMM for class {} already fitted.".format(motionclass.name))
                    continue
                
                model = self.reset_model(motionclass)
                all_class_examples = data[motionclass]
                
                inputs = []
                for idx, mocap in enumerate(all_class_examples):
                    if (motionclass, idx) == left_out:
                        continue
                    h = self.compute_h(mocap)
                    inputs.append(h)
                
                X = np.concatenate(inputs)
                lengths = [len(h) for h in inputs]

                if self.verbose:
                    print("Fitting HMM for class {} with {} samples...".format(motionclass.name, len(inputs)), end="")
                model.fit(X, lengths)
                if self.verbose:
                    print("done")

            if cross_validate:
                previous_left_out_class = left_out[0]
                pred = self.predict(data[left_out[0]][left_out[1]], show_stats=self.verbose)
                results.append(pred[0], left_out[0])
                if self.verbose:
                    print("Overall Accuracy: {:.1f}% ({}/{})".format(results.accuracy*100, results._correct, results._total))
                yield results

        yield results
                


    def train(self, data: Data, cross_validate=False, generate=False):
        """
        Fits the HMMs based on each class's data.

        If cross_validate=True, Leave One Out Cross Validation (LOOCV) is performed. For each CV iteration only the GMM of the affected class (the one from which the example mocap was left out) is refitted to improve speed.

        If generate=True an iterator (generator) is returned that produces the intermediate cross validation results.

        The final model is always trained on all available data.

        Returns:
            A PredictionResults object containing the results of the cross_validation or a generator.
        """
        gen = self._train_generator(data, cross_validate)
        if generate:
            return gen
        else:
            return list(gen)[-1]
                

    def predict(self, mocap: MotionCapture, show_stats=False):
        """
        Uses the model to predict the MotionClass of the MotionCapture object
        Returns:
            pred: The predicted MotionClass of the mocap
            confidence: A float roughly indicating the confidence of the prediction (based on the normalized difference between the scores of the first and the second guess)
        """
        
        h = self.compute_h(mocap)
        scores = []
        for motionclass in self.hmms:
            m = self.hmms[motionclass]
            try: # some score values are -inf causing an exception
                scores.append((m.score(h), motionclass))
            except:
                pass
        max_idx = max(enumerate(scores),key=lambda x: x[1])[0]

        max_score = scores[max_idx][0]
        diffs = [(max_score-s, n) for (s,n) in scores]
        max_diff = max(diffs)[0]
        norm_diffs = sorted([(d/max_diff, n) for (d,n) in diffs])
        confidence = norm_diffs[1][0]

        pred = scores[max_idx][1]

        if show_stats:
            print("Normalized Score Differences:\n")
            for (nd, mc) in norm_diffs:
                print("{}: {}".format(mc.name, nd))
            print("\nPrediction: {} ({:.1f}% confidence)".format(pred, confidence*100))            
                    
        return(pred, confidence)


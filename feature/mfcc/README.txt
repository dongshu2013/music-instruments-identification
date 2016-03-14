
The filename indicates its contents
For example: 
"mfcc_dim10x13_training_norm.npz" contians MFCC features of training data, 
each sample's feature is 10*13 (unrolled to 1*130) dimensional and is normalized


HOW TO USE:

data = numpy.load("mfcc_dim10x13_training_norm.npz")
features = data['features']
labels = data['labels']
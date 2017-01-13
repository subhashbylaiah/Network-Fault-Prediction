import data_preprocessing
import model
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    print "Processing Data....."
    data_preprocessing.main()

    print ("Running Classification Algorithms...")
    model.main()
    print ("Done!")

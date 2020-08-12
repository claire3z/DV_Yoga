dl data:
 - in download_data run "dlData(savePath = ...)"
 - if u want to change the size of the images, plot some images or dl only part of the data use
	"dlData(savePath = ..., sizeImageInterpolated = ..., numPlotImages = ..., numImages = ...)"
 - image size can be changed but 100 is used in this project

preprocess:
 - in preprocess_data run "splitTrainEvalSet(train = 0.8,eva = 0.2, path = ..., shuffle = False)" to split
 - run
	"addKeypoints(sizeImageInterpolated, path, useTestset = False, saveImages = 0)"
	"keysToTensor(path, useTestset = False)"
	"getKeypointsInFormOfImages(path, useTestset = False)"
 to get keypoints and construct them in the way used in the models
 run with useTestset = True to generate the testset

train:
 - in main run "data, labels_, dataTest, labelsTest, poseTrain, poseTest, poseTrain1, poseTest1 = loadData(path = ...)"
 to load the data
 - to train run "train(modelName = "...", learning_rate = ..., num_epochs = ..., batch_size = ..., dropout = ...,
         reproducibility = True, buildGraphs = True)"
 the model names are: cnn, hier (for hierachical model), combined1, combined2, pose, dense, dense_hier, 
	dense_combined1, dense_combined2

Testset:
 - in run_testset_on_models run 
	"getTestAccAndLoss(modelpath = "....", modelName = "...", datapath = "...")"
 to get the testset accuracy
 - run "printNumPara()" to get the parameter of each model



It takes long to dl all the data. If you want to run the model on the entire dataset but do not want to dl everything
I can probably send you the dataset (but its about 9GB)
### Face Recognition Assignment

* Data Set: LFW Dataset 
* Pretrained model: InceptionResnetV1

How to create a custom dataset

1. Select the people you want to detect the faces of
2. Collect at least 20 pictures for each of them
3. Align all the images using face align-reference (Assignment of Session 3

Model Traning
* Load the pretrained model and only train the last layers for the custom faces

Below is the loss curve for 500 epochs

![Loss Curve](https://github.com/prarthananbhat/tsai-phase2/blob/master/Session-04/Loss_curve.png?raw=true)

Prediction
1. Deployed the Model on AWS Lambda

Try on your own here: https://master.d3rfeydp6bpq4r.amplifyapp.com/





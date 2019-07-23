
Steps:
--------

1. Go to three_datasets folder.

2. Go to any of the folder [chicago-bike, flight-delay or youtube]. This readme assumes that you have chosen chicago-bike.

3. Note that, "bike-sampled.csv" file in the "data" folder has been zipped for uploading into the github repo. You have to unzip it first. 

4. Execute chicago-bike.py. This will create two files: "ChicagoBike.txt" and "ChicagoBike.json". Each line of ChicagoBike.txt is a prediction task and each line of "ChicagoBike.json" is the corresponding json string object of that prediction task.

5. Execute rewrite.py. It will create a more human readable version of the prediction task descriptions.
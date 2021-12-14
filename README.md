# TagupMLChallenge
This was a challenge for the Tagup Data Engineering position. It required that an SQL database used as an input was to be processed to filter the data and output an array of the final data. 

The code is commented to help walk through the process but I will also dive into it here. The SQL database is opened with a SQLite3 connection with the individual data tables read into Pandas DataFrames. The DataFrames are then combined into a single Frame for ease of use in the statistical analysis. A measurement was taken called a z-score that looks at a value's relationship to the average of the dataset. Any value with a z-score greater than 3 was marked to be removed from the dataset. A data point that has a high number of standard deviations are called outliers and can skew a dataset in the wrong direction. The DataFrames of booleans marked with False were combined together using an AND logic statement so that if any of the 4 features were greater than the threshhold then the whole data point was removed. This brough the dataset down from 60,000 entries to 49,141. Further updates to this processing pipeline would include data normalization to adjust the values between 0 and 1 as most Machine Learning techniques that I have encountered greatly prefer values in this range.

The next step in this pipeline was to add in the static data that was generated by ExampleCo. The method that was used is currently very slow but gets the job done. This would be a focus point for any areas of improval. It works by checking the Combined DataFrame of all of the feature sets for the metadata that contains the machine_id. A second DataFrame is then created from the static_data that copies the extra metadata for each machine_id. The two DataFrames are then combined into a single DataFrame. Following this, the DataFrame is translated into a Numpy Array for saving. 

Two extra functions were also created in this project. The first one is a simple plotting tool to help visualize the change in data before and after the outliers were removed. The graphs of the 4 feature sets are shown below. Notice the difference not only in length of the array (as 11,000 points were removed), but also the change in average value. The blue lines signify the unfiltered data with much more noise between points, and the filtered data in orange. 

The second function is a way to upload the data to AWS S3. It is currently setup to upload an array that was saved to disk, but can be changed to pull directly from the Python script, removing the overhead of having to save each run to a disk first. To get this function to work, the function must be edited to include AWS login information. 

![Feature_0](https://user-images.githubusercontent.com/55160277/145913054-6f3e4c8a-f0e7-48a9-a738-0ea29c6d889f.png)
![Feature_1](https://user-images.githubusercontent.com/55160277/145913059-41183efd-69d3-4481-8eed-974a3a24c7bc.png)
![Feature_2](https://user-images.githubusercontent.com/55160277/145913063-cc8411cc-1811-4583-a213-c4d316568f41.png)
![Feature_3](https://user-images.githubusercontent.com/55160277/145913070-068d4422-bde2-46fe-b744-c291e6a373a1.png)




## Installation
1. Download repository from GitHub.
2. Extract folder to directory on device.
3. Run script either on command line or in IDE. 

## Required Packages!

1. Pandas
2. SQLite3
3. Scipy
4. Numpy
5. Boto3
6. Logging
7. OS
8. Matplotlib

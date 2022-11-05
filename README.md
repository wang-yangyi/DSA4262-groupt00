# DSA4262 groupt00

Final project for DSA4262. We aimed to develop a method to first identify RNA modifications to train the model accounting for any inaccuracies followed by a prediction model to determine m6a sites in direct RNA sequencing samples.

## Installation

Use the github repository https://github.com/wang-yangyi/DSA4262-groupt00.git
Within the ubuntu instance, run the following commands below to pull the github repo into your ubuntu instance and the required packages to run the model:

```bash
git clone https://github.com/wang-yangyi/DSA4262-groupt00.git
sudo apt install python3-pip
mkdir workshop
mkdir workshop/data

aws s3 cp --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/SGNex_Hct116_directRNA_replicate4_run3/data.json workshop/data

cd DSA4262-groupt00

pip install -r requirements.txt

```

To enter python, run the following commands below:

```bash
python3
```

## Usage

Within python:

1. To import our model and run the training of the model, run the following commands below:

```python
import T00

# '../data/data.json' is the file path to data.json train file 
# '../data/data.info' is the file path to data.info train file 
T00.T00_model().train_model('../data/data.json', '../data/data.info')

```

2. To import our model and test our model against the SGnex data, run the following commands below:

```python
import T00

# '../workshop/data.json' is the file path to SGnex data you pulled from the S3
T00.T00_model().test_SGnex_data('../workshop/data/data.json')

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[T00]

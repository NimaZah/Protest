import re
# imports for notebook boilerplate
!pip install -Uqq fastbook
import fastbook
from fastbook import *
from fastai.tabular.all import *
fastbook.setup_book()

path=Path('data/')
path
#downloading data
path.mkdir(parents=True, exist_ok=True)
path.ls()

# loading the data into a dataframe
df = pd.read_csv('https://raw.githubusercontent.com/joankovacs/mass_mobilization/main/cleaned_protests.csv')
#check the first rows of the dataframe
df.head()
df.drop(['sources', 'notes', 'protesteridentity'], axis=1, inplace=True)
df.head()

#print the unique values of duration and assign them to unique.
unique = df['protest_length'].unique()
# sort the values of unique.
unique.sort()
unique

# create a function to assign a category to the values of protest_length column. The function will check if the value 
# of the protest_length is less than or equal to 30, and if it is, then it returns '1 months'. If the value of the
# protest_length column is greater than 30 but less than or equal to 90, then it returns '1-3 months'. If 
# value of the protest_length column is greater than 90 but less than or equal to 180, then it returns
# '1-6 months'. If the value of the protest_length column is greater than 180, then it returns '>6 months'

def protest_length_category(protest_length):
    if protest_length <= 30:
        return '1 month'
    elif protest_length > 30 and protest_length <= 90:
        return '1-3 months'
    elif protest_length > 90 and protest_length <= 180:
        return '3-6 months'
    elif protest_length > 180:
        return '>6 months'
        
# Apply the function to the protest_length column.
# df['protest_length_category'] = df.apply(protest_length_category, axis=1)
# df['protest_length_category']
# df.head()
df['protest_length_category'] = df['protest_length'].apply(protest_length_category)
df.head()

df['protest_length_category'].unique()

df.drop(['protest_length', 'Unnamed: 0', 'Unnamed: 0'], axis= 1, inplace= True)
df.head()

## OpenAI Section

# export the dataframe to the local machine.
from google.colab import files
df.to_csv('cleaned_protests.csv') 
files.download('cleaned_protests.csv')

# create a directory for the csv file in colab.
!mkdir -p protests
from google.colab import files
uploaded= files.upload()
# move the file to its directory.
!mv cleaned_protests.csv protests/
# associate path with the directory
path = Path('protests')
# ingest the dataset into the df_train DataFrame
df_train=pd.read_csv(path/'cleaned_protests.csv')
df_train.head()

# get a count by column of missing values
count = df_train.isna().sum()
df_train_missing = (pd.concat([count.rename('missing_count'),
                     count.div(len(df_train))
                          .rename('missing_ratio')],axis = 1)
             .loc[count.ne(0)])

df_train_missing

# define TabularDataLoaders object using the dataframe, the list of pre-processing steps, 
# the categorical and continuous column lists
# procs = [Categorify, FillMissing, Normalize]
# valid_idx = range(len(df_train)-2000, len(df_train))
# dep_var = 'protest_length_category'
# cat_names = ['country', 'region', 'protest_length_category']
# cont_names = ['protest_length']

procs = [FillMissing,Categorify]
dep_var = 'protest_length_category'
cont,cat = cont_cat_split(df_train, 1, dep_var=dep_var) 
print("continuous columns are: ",cont)
print("categorical columns are: ",cat)

procs = [FillMissing,Categorify, Normalize]
dls = TabularDataLoaders.from_df(df_train,path,procs= procs, 
                                 cat_names= cat, cont_names = cont, 
                                 y_names = dep_var, 
                                 valid_idx=list(range((df_train.shape[0]-2000),df_train.shape[0])), bs=64)
                                 
# display a sample batch
dls.valid.show_batch()

# define and fit the model
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(3)

# show the loss function used by the learner
learn.loss_func

# show a set of results from the model
learn.show_results()

learn.summary()
learn.path
learn.path = path

# pickle the trained model and save it to the hard disk.
import  pickle
file_path = learn.path/'trained_model.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(learn, f)

# download the model from google colab to the local machine.
from google.colab import files
files.download('trained_model.pkl')
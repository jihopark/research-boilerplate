# boilerplate codes related to file loading

###############
#Pandas
#import pandas as pd
###############

# for jupyter notebook
# pd.set_option('display.max_colwidth', -1)


# loading from csv
"""
df = pd.read_csv('./%s' % filename,
                     sep="\t",
                     header=None,
                     skiprows=[0],
                     names=["Tweet_ID", "Text", "Previous"],
                     error_bad_lines=False)
"""

# loading from numpy array
# df2 = pd.DataFrame({"time": tmp[:,0], "writer": tmp[:, 1], "text": tmp[:, 2]})

# concatenating dataframes
# pd.concat([df1, df2])

# df.drop_duplicates()



# gather value counts for each label
# df['Tweet_ID'].value_counts()

###############
#Codecs
#import codecs
###############

# with codecs.open(filename,"r", "utf-8") as f:


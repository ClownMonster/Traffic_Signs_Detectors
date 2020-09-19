'''
Analysing the images in the dataset

Dataset has 43 classes i,e 43 different types of traffic signs each intern has many images,
a total of 39209 images are in dataset

'''

import seaborn as sns

fig = sns.distplot(output, kde=False, bins = 43, hist = True, 
                    hist_kws=dict(edgecolor="black", linewidth=2))
fig.set(title = "Traffic signs frequency graph",
        xlabel = "ClassId",
        ylabel = "Frequency")


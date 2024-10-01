# Firsly we have the bags:
# s1 = {the bag of words representation becomes less parsimonious}
# s2 = {if we do not stem the words}

# We can turn them into vectors:
# We will define a library of words:
# lib = ['the', 'bag', 'of', 'words', 'representation', 'becomes', 'less', 'parsimonious', 'if', 'we', 'do', 'not', 'stem']

# v1 = [1,1,1,1,1,1,1,1,0,0,0,0,0]
# v2 = [1,0,0,1,0,0,0,0,1,1,1,1,1]

# we can now apply jaccard similarity:
# f11 = 2
# f10 = 6
# f01 = 5


# jaccard similarity = f11 / (f11 + f10 + f01) = 2 / (2 + 6 + 5) = 2 / 13 = 0.1538














#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data from URL
url = "https://jse.amstat.org/datasets/fishcatch.dat.txt"
data = pd.read_csv(url, delim_whitespace=True, skiprows=1, header=None, na_values=['?'])

data.columns = ["Obs", "Species", "Weight", "Length1", "Length2", "Length3", "Height%", "Width%", "Sex"]

# Define species mapping
species_mapping = {
    1: "Lahna",
    2: "Siika",
    3: "Saerki",
    4: "Parkki",
    5: "Norssi",
    6: "Hauki",
    7: "Ahven"
}

data["Species"] = data["Species"].map(species_mapping)
data["Sex"].fillna(np.random.choice([0, 1]), inplace=True)

# Calculate average weight of species 1
avg_weight_species_1 = data[data["Species"] == "Lahna"]["Weight"].mean()

# Fill missing values in Weight column with average weight of species 1
data["Weight"].fillna(avg_weight_species_1, inplace=True)

# Convert to CSV
data.to_csv("fish_data.csv", index=False)

# Print first 159 rows and 9 variables
print(data.head(159))

# Calculate and print average weight, height, Length1, Length2, and Length3 of each species
avg_stats = data.groupby("Species").agg({
    "Weight": "mean",
    "Length1": "mean",
    "Length2": "mean",
    "Length3": "mean",
    "Height%": "mean"
})
print("\nAverage statistics for each species:")
print(avg_stats)

# Calculate average values for each category
avg_weight = data["Weight"].mean()
avg_length1 = data["Length1"].mean()
avg_length2 = data["Length2"].mean()
avg_length3 = data["Length3"].mean()
avg_height = data["Height%"].mean()

# Count fishes below and above average for each category
below_avg_weight = (data["Weight"] < avg_weight).sum()
above_avg_weight = (data["Weight"] > avg_weight).sum()
below_avg_length1 = (data["Length1"] < avg_length1).sum()
above_avg_length1 = (data["Length1"] > avg_length1).sum()
below_avg_length2 = (data["Length2"] < avg_length2).sum()
above_avg_length2 = (data["Length2"] > avg_length2).sum()
below_avg_length3 = (data["Length3"] < avg_length3).sum()
above_avg_length3 = (data["Length3"] > avg_length3).sum()
below_avg_height = (data["Height%"] < avg_height).sum()
above_avg_height = (data["Height%"] > avg_height).sum()

# Calculate percentages
total_count = len(data)
percent_below_avg_weight = (below_avg_weight / total_count) * 100
percent_above_avg_weight = (above_avg_weight / total_count) * 100
percent_below_avg_length1 = (below_avg_length1 / total_count) * 100
percent_above_avg_length1 = (above_avg_length1 / total_count) * 100
percent_below_avg_length2 = (below_avg_length2 / total_count) * 100
percent_above_avg_length2 = (above_avg_length2 / total_count) * 100
percent_below_avg_length3 = (below_avg_length3 / total_count) * 100
percent_above_avg_length3 = (above_avg_length3 / total_count) * 100
percent_below_avg_height = (below_avg_height / total_count) * 100
percent_above_avg_height = (above_avg_height / total_count) * 100

# Print results
print("Percentage of fishes below and above average for each category:")
print("Weight:")
print("  - Below average: {:.2f}%".format(percent_below_avg_weight))
print("  - Above average: {:.2f}%".format(percent_above_avg_weight))
print("Length1:")
print("  - Below average: {:.2f}%".format(percent_below_avg_length1))
print("  - Above average: {:.2f}%".format(percent_above_avg_length1))
print("Length2:")
print("  - Below average: {:.2f}%".format(percent_below_avg_length2))
print("  - Above average: {:.2f}%".format(percent_above_avg_length2))
print("Length3:")
print("  - Below average: {:.2f}%".format(percent_below_avg_length3))
print("  - Above average: {:.2f}%".format(percent_above_avg_length3))
print("Height%:")
print("  - Below average: {:.2f}%".format(percent_below_avg_height))
print("  - Above average: {:.2f}%".format(percent_above_avg_height))

# Calculate the overall likelihood of catching a fish below or above average
likelihood_below_avg = (percent_below_avg_weight + percent_below_avg_length1 + percent_below_avg_length2 + percent_below_avg_length3 + percent_below_avg_height) / 5
likelihood_above_avg = (percent_above_avg_weight + percent_above_avg_length1 + percent_above_avg_length2 + percent_above_avg_length3 + percent_above_avg_height) / 5

print("Overall likelihood of catching a fish:")
print("  - Below average: {:.2f}%".format(likelihood_below_avg))
print("  - Above average: {:.2f}%".format(likelihood_above_avg))

# Function to simulate catching fish
def catch_fish(likelihood_below_avg, likelihood_above_avg, num_attempts):
    below_avg_count = 0
    above_avg_count = 0
    
    for _ in range(num_attempts):
        # Generate a random number between 0 and 100
        rand_num = random.uniform(0, 100)
        
        # Determine if the fish caught is below or above average
        if rand_num < likelihood_below_avg:
            below_avg_count += 1
        elif rand_num < (likelihood_below_avg + likelihood_above_avg):
            above_avg_count += 1
    
    return below_avg_count, above_avg_count

num_attempts = 1000
below_avg_count, above_avg_count = catch_fish(likelihood_below_avg, likelihood_above_avg, num_attempts)

# Calculate percentages
percent_below_avg_actual = (below_avg_count / num_attempts) * 100
percent_above_avg_actual = (above_avg_count / num_attempts) * 100

# Print the actual results
print("Actual results based on simulation:")
print("  - Below average: {:.2f}%".format(percent_below_avg_actual))
print("  - Above average: {:.2f}%".format(percent_above_avg_actual))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Initialize the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the prepared dataset to CSV
data.to_csv("fish_data_with_labels.csv", index=False)


# In[ ]:





import random

# Create a variable to store the day of the week

day = input("What day is it? ")

# Creating a dictionary to store the days of the week and the meals for each day
meals = {
    "Monday": ['Chicken', 'Rice', 'Beans'],
    "Tuesday": ['Beef', 'Rice', 'Beans'],
    "Wednesday": ['Fish', 'Rice', 'Beans'],
    "Thursday": ['Chicken', 'Rice', 'Beans'],
    "Friday": ['Beef', 'Rice', 'Beans'],
    "Saturday": ['Fish', 'Rice', 'Beans'],
    "Sunday": ['Chicken', 'Rice', 'Beans']
}

# Creating a variable to store a random number to select a meal
random_number = random.randint(0, 3)

# Creating a variable that uses the random number to select a meal
meal = meals[day][random_number]

# Printing the meal
print(meal)
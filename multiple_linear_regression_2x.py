import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Multiple linear regression involves predicting
# a dependent variable based on two or more
# independent variables. It assumes a linear
# relationship between the dependent variable
# and each independent variable. The regression
# equation for multiple linear regression
# can be represented as follows:
# Y = b0 + b1X1 + b2X2 + ... + bn*Xn

# NOTES:
# - Always use Procedural Paradigm
# - Remember to preprocess your data,
#   handle missing values, and perform feature scaling
#   if necessary before applying the regression model.

def main():
    print("\nLoading the dataset.")
    filename = "datasets/egg_production_2x.csv"
    dataset = np.genfromtxt(filename, delimiter=",", skip_header=1)
    
    print("\nPreprocessing the dataset.")
    dataset = dataset[1:]
    
    target = dataset[:, -1]     # dependent variable
    features = dataset[:, 1:-1] # independent variables
    
    # Split the dataset for testing and validation
    features1, features2, target1, target2 = train_test_split(features, target, test_size=0.2)
    
    # train the egg count model
    print("\nTraining the model.")
    egg_count_model = LinearRegression()
    egg_count_model.fit(features1, target1)

    coefficients = egg_count_model.coef_
    intercept = egg_count_model.intercept_
    
    regression_model = format_regression_model(intercept, coefficients)
    print(f"\nRegression Model: {regression_model}")
    
    # Include all from the original dataset to check accuracy with unseen data
    r_squared = egg_count_model.score(features, target)
    r_squared_percent = r_squared*100
    
    print(f"\nAbout {r_squared_percent:.2f}% of the data can be explained by the regression plane y = {regression_model}")
    
    show_plot(egg_count_model, features, target)
    
    predict(egg_count_model)

def predict(egg_count_model):
    while 1:
        print("\n[Predict Egg Count]")
        try:
            layers = int(input(" Layer head count: ").strip())
            temperature = float(input(" Temperature (C): "))
        except:
            print("\nInvalid input, please input a valid number.")
            break
        
        prediction = int(egg_count_model.predict([[layers, temperature]])[0])
        print(f"Prediction: {prediction}")

def format_regression_model(intercept, coefficients, precision=1):
    formula = []
    counters = range(1, len(coefficients)+1)
    
    formula.append(str(round(intercept, precision)))
    for coefficient,counter in zip(coefficients, counters):
        label = f"x{counter}"
        if coefficient >= 0:
            coefficient = round(coefficient, precision)
            formula.append(f"+ {coefficient}*{label}")
            continue
        coefficient = abs(round(coefficient, precision))
        formula.append(f"- {coefficient}*{label}")

    return " ".join(formula)

def show_plot(egg_count_model, features, target):
    
    layers = features[:, 0]
    temperature = features[:, 1]
    
    # Generate an `N` evenly spaced numbers
    #   inclusively between minimum and maximum
    range1 = np.linspace(min(layers), max(layers), num=10)
    range2 = np.linspace(min(temperature), max(temperature), num=10)
    
    # Generate arrays of coordinates based on the input
    grid1, grid2 = np.meshgrid(range1, range2)
    
    # Create predictions based on the meshgrid outputs
    #   Each meshgrid output is flattened before processed to a numpy array.
    #   `.T` gets the transposed (rotated) array
    new_dataset = np.array([grid1.flatten(), grid2.flatten()]).T

    predictions1 = egg_count_model.predict(features)
    predictions2 = egg_count_model.predict(new_dataset)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    # marker styles: circles (o), squares (s), triangles (v), and crosses (+)
    ax.scatter(layers, temperature, target, color="b", marker="o")
    ax.scatter(layers, temperature, predictions1, color="r", marker="o", alpha=0.3)
    
    # Plot the regression plane
    ax.plot_surface(grid1, grid2, predictions2.reshape(grid1.shape), color="gray", alpha=0.5)
    
    plt.title("Multiple Linear Regression (2x)")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Temperature")
    ax.set_zlabel("Egg Count")
    
    plt.show()

if __name__ == "__main__":
    main()
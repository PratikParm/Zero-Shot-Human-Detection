def alert(prediction):
    """
    Function to alert the user based on the prediction.
    """
    if prediction != "other":
        print(f"Alert: {prediction.title()} detected!")
        return
    else:
        return
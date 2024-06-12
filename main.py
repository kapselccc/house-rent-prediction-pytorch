import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

test_run = 1


class Model(nn.Module):
    def __init__(self, input_num: int, layers: list[int], dropout_p: float = 0.5):
        super(Model, self).__init__()

        layers_list = []

        in_num = input_num
        for i in layers:
            layers_list.append(nn.Linear(in_num, i))
            layers_list.append(nn.ReLU(inplace=True))
            layers_list.append(nn.BatchNorm1d(i))
            layers_list.append(nn.Dropout(dropout_p))
            in_num = i

        layers_list.append(nn.Linear(layers[-1], 1))
        self.layers = nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    data = pd.read_csv('House_Rent_Dataset.csv')

    # Change data
    data.drop(columns='Area Locality', inplace=True)
    data.drop(columns='Posted On', inplace=True)
    data["Total Floors"] = data["Floor"].apply(lambda floor: floor.split()[-1]).replace('Ground', 1)
    data["Floor"] = (data["Floor"].apply(lambda floor: floor.split()[0])
                     .replace('Ground', 0)
                     .replace('Upper', 99)
                     .replace('Lower', -1)).astype("int64")

    data = pd.get_dummies(data, columns=['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of '
                                                                                                       'Contact'])
    # Split data
    X = data.drop(columns='Rent', axis=1).astype("int64")
    y = data['Rent'].astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=77)

    X_train = torch.FloatTensor(X_train.to_numpy())
    X_val = torch.FloatTensor(X_val.to_numpy())
    X_test = torch.FloatTensor(X_test.to_numpy())
    y_train = torch.FloatTensor(y_train.to_numpy())
    y_test = torch.FloatTensor(y_test.to_numpy())
    y_val = torch.FloatTensor(y_val.to_numpy())

    # train model
    torch.manual_seed(77)

    model = Model(X.shape[1], [200, 100], 0.4)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 400
    losses = []
    start_time = time.time()
    for i in range(epochs):
        i += 1
        for x in X_train:
            y_pred = model.forward(x)
            loss = torch.sqrt(criterion(y_pred, y_train))
            losses.append(loss.item())


        if i % 10 == 0:
            print(f"Epoch ({i:3}): Loss: {loss.item():10.5f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final loss after {epochs} epochs: {losses[-1]:8.5f}")
    print(f"Duration: {time.time() - start_time:.2f} seconds")

    plt.plot(range(epochs), losses)
    plt.ylabel('RMSE Loss')
    plt.xlabel('epoch')

    # Validation
    with torch.no_grad():
        y_pred = model(X_val)
        loss = torch.square(criterion(y_pred, y_val))
        print(f"Validation loss after {epochs} epochs: {loss:8.5f}")

    torch.save(model.state_dict(), f'Model_{test_run}.pt')

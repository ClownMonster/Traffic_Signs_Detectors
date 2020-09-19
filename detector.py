import cv2
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential


def resize_img(img):
    '''
    since CNN model can not be trained on images of different size all the training data
    needs to be resized into the same dimensions

    '''
    return cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)




# loading the data

list_images = []
output = []
for dir in os.listdir(data_dir):
    if dir == '.DS_Store' :
        continue
    
    inner_dir = os.path.join(data_dir, dir)
    csv_file = pd.read_csv(os.path.join(inner_dir,"GT-" + dir + '.csv'), sep=';')
    for row in csv_file.iterrows() :
        img_path = os.path.join(inner_dir, row[1].Filename)
        img = imread(img_path)
        img = img[row[1]['Roi.X1']:row[1]['Roi.X2'],row[1]['Roi.Y1']:row[1]['Roi.Y2'],:]
        img = resize_cv(img)
        list_images.append(img)
        output.append(row[1].ClassId)


# spliting the data

input_array = np.stack(list_images)
train_y = keras.utils.np_utils.to_categorical(output)randomize = np.arange(len(input_array))
np.random.shuffle(randomize)
x = input_array[randomize]
y = train_y[randomize]


split_size = int(x.shape[0]*0.6)
train_x, val_x = x[:split_size], x[split_size:]
train1_y, val_y = y[:split_size], y[split_size:]split_size = int(val_x.shape[0]*0.5)
val_x, test_x = val_x[:split_size], val_x[split_size:]
val_y, test_y = val_y[:split_size], val_y[split_size:]

# training the model

hidden_num_units = 2048
hidden_num_units1 = 1024
hidden_num_units2 = 128
output_num_units = 43

epochs = 10
batch_size = 16
pool_size = (2, 2)
# list_images /= 255.0
input_shape = Input(shape=(32, 32,3))

# model layers
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64,64,3), padding='same'),
    BatchNormalization(),Conv2D(16, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.2),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.2),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.2),Flatten(),Dense(units=hidden_num_units, activation='relu'),
    Dropout(0.3),
    Dense(units=hidden_num_units1, activation='relu'),
    Dropout(0.3),
    Dense(units=hidden_num_units2, activation='relu'),
    Dropout(0.3),
    Dense(units=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', 
                optimizer=Adam(lr=1e-4), metrics=['accuracy'])

trained_model_conv = model.fit(train_x.reshape(-1,64,64,3),
                                train1_y, epochs=epochs, 
                                batch_size=batch_size, 
                                validation_data=(val_x, val_y)


# model Evalutaion
model.evaluate(test_x, test_y)

# predicting result
pred = model.predict_classes(test_x)

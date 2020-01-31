# network with 4 convolutional layers (from 512 to 64) with BatchNormalization and 1*3+1 dense, LR with scheduler (starting from 1e-3) and Adam optimizer
 
###Previous attempt: _network with 4 convolutional layers (from 512 to 64) and 1*3+1 dense, LR with scheduler and Adamax optimizer_
With all possible LR values the model is not good cause of Adamax optimizer: training loss went to NaN immediately.

## RMSE = ?, Validation Loss = ?

_here is the NN used:_

	def CNN_galaxy():

	    model = Sequential()
	    model.add(Conv2D(512, (3, 3), input_shape=(224, 224, 3)))
	    model.add(Conv2D(256, (3, 3)))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(256, (3, 3)))
	    model.add(Conv2D(128, (3, 3)))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(128, (3, 3)))
	    model.add(Conv2D(64, (3, 3)))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(64, (3, 3)))
	    model.add(Conv2D(64, (3, 3)))
	    model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(GlobalMaxPooling2D())


	    model.add(Dropout(0.25))
	    model.add(Dense(64))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.25))
	    model.add(Dense(64))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.25))
	    model.add(Dense(64))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.25))
	    model.add(Dense(NUM_CLASSES))
	    model.add(Activation('sigmoid'))

	    for layer in model.layers:
	      layer.trainable = True

	    optimizer = keras.optimizers.Adam(lr=LR, decay=WEIGHT_DECAY)
	    model.compile(optimizer, loss='binary_crossentropy', metrics=[root_mean_squared_error])

	    return model

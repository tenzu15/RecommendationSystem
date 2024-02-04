import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelEncoder
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth




with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

#name, authentication_status, username = authenticator.login('Login', 'main')
authenticator.login()


if st.session_state["authentication_status"]:
    authenticator.logout()
    #st.write(f'Welcome *{name}*')
    #st.title('Some content')

    st.title("Movie Recommendations")


    #st.title('Uber pickups in NYC')

    #DATE_COLUMN = 'date/time'
    DATA_URL = ("refined_dataset_final.csv")


    @st.cache_data
    def load_data():
        data = pd.read_csv(DATA_URL)
        #lowercase = lambda x: str(x).lower()
        #data.rename(lowercase, axis='columns', inplace=True)
        #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data

        
    def recommender_system(user_id, model, n_movies, refined_dataset):

        user_enc = LabelEncoder()
        refined_dataset['user'] = user_enc.fit_transform(refined_dataset['userId'].values)
        n_users = refined_dataset['user'].nunique()
            

        item_enc = LabelEncoder()
        refined_dataset['movie'] = item_enc.fit_transform(refined_dataset['title'].values)
        n_movies = refined_dataset['movie'].nunique()
            

        refined_dataset['rating'] = refined_dataset['rating'].values.astype(np.float32)
        min_rating = min(refined_dataset['rating'])
        max_rating = max(refined_dataset['rating'])
        n_factors = 150
        #n_users, n_movies, min_rating, max_rating = 610, 9719, 0.5, 5.0

        print("")
        st.write("Movie seen by the User:")
        st.write(list(refined_dataset[refined_dataset['userId'] == user_id]['title']))
        st.write("")

        encoded_user_id = user_enc.transform([user_id])
        n_movies=5
        seen_movies = list(refined_dataset[refined_dataset['userId'] == user_id]['movie'])
        unseen_movies = [i for i in range(min(refined_dataset['movie']), max(refined_dataset['movie'])+1) if i not in seen_movies]
        model_input = [np.asarray(list(encoded_user_id)*len(unseen_movies)), np.asarray(unseen_movies)]
        predicted_ratings = model.predict(model_input)
        predicted_ratings = np.max(predicted_ratings, axis=1)
        sorted_index = np.argsort(predicted_ratings)[::-1]
        recommended_movies = item_enc.inverse_transform(sorted_index)
        print("---------------------------------------------------------------------------------")
        st.write("Top "+str(n_movies)+" Movie recommendations for the User "+str(user_id)+ " are:")
        st.write(list(recommended_movies[:n_movies]))

    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text("Done!")


    #model structure
    def get_model():
        n_factors = 150
        n_users, n_movies, min_rating, max_rating = 610, 9719, 0.5, 5.0
        ## Initializing a input layer for users
        user = tf.keras.layers.Input(shape = (1,))

        ## Embedding layer for n_factors of users
        u = keras.layers.Embedding(n_users, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer = tf.keras.regularizers.l2(1e-6))(user)
        u = tf.keras.layers.Reshape((n_factors,))(u)

        ## Initializing a input layer for movies
        movie = tf.keras.layers.Input(shape = (1,))

        ## Embedding layer for n_factors of movies
        m = keras.layers.Embedding(n_movies, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(movie)
        m = tf.keras.layers.Reshape((n_factors,))(m)

        ## stacking up both user and movie embeddings
        x = tf.keras.layers.Concatenate()([u,m])
        x = tf.keras.layers.Dropout(0.05)(x)

        ## Adding a Dense layer to the architecture
        x = tf.keras.layers.Dense(32, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(0.05)(x)

        x = tf.keras.layers.Dense(16, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Dropout(0.05)(x)

        ## Adding an Output layer with Sigmoid activation funtion which gives output between 0 and 1
        x = tf.keras.layers.Dense(9)(x)
        x = tf.keras.layers.Activation(activation='softmax')(x)
        model = tf.keras.models.Model(inputs=[user,movie], outputs=x)
        model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        return model


    model=get_model()
    model.load_weights("model.h5")

    title = st.text_input('User Id', '0')
    st.write('Enter your user id for recommendation', title)

    try:
        print(recommender_system(int(title), model, 5, data))
    except:
        print("")

    st.write("##")
    tab1 ,tab2, tab4, tab3, tab5 = st.tabs(["About","Ideology","Feature Enhancement","Original Dataset", "DataSplit"])

    with tab1:
        st.caption('This a DNN Based Movie Recommendation System')
        #st.caption('In upcoming versions new movies would be added 😎')   #:blue[:sunglasses:]
    with tab2:
        st.caption("")
        st.write(" Due to flexibility of the input layer of network, DNNs can easily incorporate query features and item features which can help capture the specific interests of a user and improve the relevance of recommendations.")
        st.write("There are different types of Deep Neural Networks applications like DNN with Softmax layer, DNN with Autoencoder architecture or may it be Recommender System with Wide & Deep Neural Networks that can be applied to Recommender Systems for better movies to recommend.")

        st.write("For this project, Softmax Deep Neural Networks are used to recommend movies. Users and Movies are one-hot encoded and fed into the Deep Neural Network as different distinct inputs and ratings are given as output.")

        st.write("Deep Neural Netwoçrk model was built by extracting the latent features of Users and movies with the help of Embedding layers and then Dense layers with dropouts were stacked in the end and finally a Dense layer with 9 neurons (one for each possible rating from 1 to 5) with a Softmax activation function was added.")

        st.write("Hyperparmeters of the model were tuning, many loss functions and optimizers were tried with minimum validation loss as metric to built the model and get the weights.")

        st.write("Finally, 'SGD' for optimizer and Sparse Categorical Cross entropy for loss function were picked.")

        st.write("Movie Recommendations:")
        st.write("User id is taken as input from the User. Then the movie ids which were not already seen by extracted from the available dataframe.")

        st.write("How this DNN model works is, it takes two inputs, one of the input has user id's and the other has corresponding movie id's. Here DNN model tries to predict the ratings of the user - movie combination. So, we can input a specific user id (broadcasting it with the size of other input) and unseen movie id of the user and expect the model to give the ratings of the movies which would have been the ratings given by the user. Here, the ratings are already normalized and as we need the movies which interest the user more, ratings are not brought back to 0-5 scale.")

        st.write("DNN model is used to predict the ratings of the unseen movies.")

    with tab4:

        st.write("Here is a modified version of the dataset:")
        df = data.copy()
        #df.drop("0")
        st.write(df.head())
        st.write(" We make changes to the dataset by combining and the 3 intial datasets based on Movie ID, followed by removing the duplicates")
        st.write(" We also make use of label encoders and normalization for enconding values.")
    with tab3:

        st.write("Initial Dataset:")
        movies=pd.read_csv("data/RecomData/movies.csv")
        ratings= pd.read_csv("data/RecomData/ratings.csv")
        st.write("Movies")
        st.write(movies.head())
        st.write("Ratings")
        st.write(ratings.head())

    with tab5:

        st.write("Training & Test Set Split:")

elif st.session_state["authentication_status"] == False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] == None:
    st.warning('Please enter your username and password')


    
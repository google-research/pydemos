# LiT Demo


## Quick steps for deployment

Last update tested using **Python 3.10.5**.

### Option 1: Manual demo build

1.  Setup virtual environment.

    ```sh
    virtualenv venv && source venv/bin/activate
    ```

2.  Install requirements.

    ```sh
    pip install -r requirements.txt
    ```

3.  Run the app on localhost.

    ```sh
    streamlit run app.py
    ```


### Option 2: Docker build

1.  Build image from the source directory.

    ```sh
    docker build -t lit-demo .
    ```

2.  Run it on localhost

    ```sh
    docker run --network="host" lit-demo
    ```
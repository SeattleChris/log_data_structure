# import config
from config import Config
import lsd


# app = application.create_app(config)
config_obj = Config()
app = lsd.create_app(config_obj)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)

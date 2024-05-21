from App.Derivative.ZController.Controller import Controller
import logging

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    Controller().run()
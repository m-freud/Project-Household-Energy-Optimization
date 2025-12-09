from dotenv import load_dotenv
import os

from config import Config

load_dotenv()  # sucht automatisch im Projektroot

token = Config.INFLUX_TOKEN
url = Config.INFLUX_URL
org = Config.INFLUX_ORG
bucket = Config.INFLUX_BUCKET

def load_household_data(household_id):
    '''
    Loads household data for the given household ID.
    Parameters:
        household_id: Identifier for the household
    '''
    # Implement data loading logic here
    pass


def generate_profile(player_id, optimizer):
    '''
    Generates a simulation profile for a given player and optimizer.
    Parameters:
        player_id: Identifier for the player
        optimizer: Optimizer settings or object
    '''



from sentinelhub import SHConfig
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

import os
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logging.captureWarnings(True)

load_dotenv()

#Setup Google Maps API Key
GOOGLE_MAPS_STATIC_API_KEY = str(os.getenv('GOOGLE_MAPS_STATIC_API_KEY'))

# Setup Copernicus client credentials
CLIENT_ID = os.getenv('COPERNICUS_CLIENT_ID')
CLIENT_SECRET = os.getenv('COPERNICUS_CLIENT_SECRET')
CONFIG_NAME = str(os.getenv('COPERNICUS_CONFIG_NAME'))

# Setup config params for Copernicus dataspace Ecosystem users
config = SHConfig()

config.sh_client_id = CLIENT_ID
config.sh_client_secret = CLIENT_SECRET
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'

config.save(CONFIG_NAME)

config

# def sentinelhub_compliance_hook(response):
#     '''
#     Check response status and raise an error if needed.
#     '''
#     response.raise_for_status()
#     return response

# # Create a session
# client = BackendApplicationClient(client_id=client_id)
# oauth = OAuth2Session(client=client)

# # Get token for the session
# token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
#                           client_secret=client_secret, include_client_id=True)

# # Register the compliance hook to check responses before sending back to user
# oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

# # All requests using this session will have an access token automatically added
# resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
# print(resp.content)

from flask import jsonify
from ..main import app

class APIError(Exception):
    status_code = 500

    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message

        if(status_code is not None):
            self.status_code = status_code
        
    def to_dict(self):
        res = dict(())
        res['message'] = self.message
        return res

@app.errorhandler(APIError)
def handleAPIError(error):
    res = jsonify(error.to_dict())
    res.status_code = error.status_code
    return res
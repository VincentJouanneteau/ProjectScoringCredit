import unittest
import requests

        
class TestRefusAcceptation(unittest.TestCase):
    def test_should_be_accepted(self):
        url_api = "http://13.39.87.160:8080/score/100001"
        response = requests.get(url_api)
        decision = response.json()['prediction_text']
        self.assertEqual(decision,"Accepté")
    
    def test_should_be_refused(self):
        url_api = "http://13.39.87.160:8080/score/100561"
        response = requests.get(url_api)
        decision = response.json()['prediction_text']
        self.assertEqual(decision,"Refusé")
        
        
if __name__ == "__main__":
    unittest.main()
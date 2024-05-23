import unittest
import requests

        
class TestRefusAcceptation(unittest.TestCase):
    # Test du client 100001 qui doit être accepté
    def test_should_be_accepted(self):
        url_local_api = "http://13.39.87.160:8080/score/100001"
        response = requests.get(url_api)
        decision = response.json()['prediction_text']
        self.assertEqual(decision,"Accepté")
    # Test du client 100561 qui doit être refusé
    def test_should_be_refused(self):
        url_local_api = "http://13.39.87.160:8080/score/100561"
        response = requests.get(url_api)
        decision = response.json()['prediction_text']
        self.assertEqual(decision,"Refusé")
        
        
if __name__ == "__main__":
    unittest.main()
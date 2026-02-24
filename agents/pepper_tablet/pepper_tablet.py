# Import basic preliminaries
import time

# import the device(s) we will be using
from sic_framework.devices import Pepper

# import the message types we will be using
from sic_framework.devices.common_pepper.pepper_tablet import (
    ClearDisplayMessage,
    UrlMessage,
)

WEBSITE_URL = "http://10.0.0.184:8000/"
DISPLAY_DURATION_URL = 8.0


class PepperTablet:
    def __init__(self, pepper: Pepper):
        self.pepper = pepper

    def run(self):
        try:
            if WEBSITE_URL:
                print("Displaying website: %s", WEBSITE_URL)
                self.pepper.tablet.send_message(UrlMessage(WEBSITE_URL))
                time.sleep(DISPLAY_DURATION_URL)
        except Exception as exc:
            print("Error during tablet demo: %s", exc)
            import traceback
            traceback.print_exc()

    def clear_screen(self):
        self.pepper.tablet.send_message(ClearDisplayMessage())


if __name__ == "__main__":
    pepper = Pepper(ip="10.0.0.168")
    demo = PepperTablet(pepper)
    demo.run()

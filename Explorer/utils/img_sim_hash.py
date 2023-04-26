from PIL import Image
import imagehash
from io import BytesIO
from appium import webdriver
from appium_helper import AppiumLauncher

def img_hash_distance(img_b,img_a):
    pil_img_a = Image.open(BytesIO(img_a))
    pil_img_b = Image.open(BytesIO(img_b))
    image_a_hash = imagehash.phash(pil_img_a)
    image_b_hash = imagehash.phash(pil_img_b)
    similarity = image_a_hash - image_b_hash
    print('similarity is ',similarity)
    return similarity

def generate_xml():
    appium = AppiumLauncher(4723)
    desired_caps = {'platformName': 'Android',
                     'platformVersion': '7',
                     'udid': 'emulator-5554',
                     'deviceName': 'Xiaomi10',
                     'autoGrantPermissions': False,
                     'fullReset': False,
                     'resetKeyboard': True,
                     'androidInstallTimeout': 30000,
                     'isHeadless': False,
                     'automationName': 'uiautomator2',
                     'adbExecTimeout': 30000,
                     'appWaitActivity': '*',
                     'newCommandTimeout': 200}

    driver = webdriver.Remote(f'http://127.0.0.1:4723/wd/hub', desired_caps)
    img_a = driver.get_screenshot_as_png()
    img_b = driver.get_screenshot_as_png()
    distance = img_hash_distance(img_b,img_a)
    print(distance)

if __name__ == '__main__':
    generate_xml()
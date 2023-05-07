## GUIMind
---
GUIMind is a novel automated tool to detect violations of the Data Minimization Principle. The tool consists of two modules: Explorer and Fidelity Checker. Explorer utilizes a reinforcement learning model to explore app activities and monitor access to sensitive APIs that require sensitive permissions. Fidelity Checker uses the existing tool [APICOG](https://ieeexplore.ieee.org/abstract/document/9251054) to check whether an activity collects more sensitive permissions than the users expect.

#### Framework:
<img src="[https://github.com/AnonymousSE202X/GUIMind/blob/main/guimind_architecture.png](https://github.com/AnonymousSE202X/GUIMind/blob/main/guimind_architecture.png)" width="500">
</div>

## Getting Started
---
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
---
* A rooted Google Pixel 5 device
* Frida version corresponding to your device if not a Google Pixel 5 device
* Python 3.x
* Required Python packages (see requirements.txt)

## Installation
---
1. Clone the GUIMind GitHub repository to your local machine.
   ```sh
   git clone https://github.com/AnonymousSE202X/GUIMind.git
   ```
2. Install required dependencies by running the following command:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the pre-trained weight models, Frida, and Standford NLP Parser from [Google Drive](https://drive.google.com/drive/folders/1FrDyUGGBHNazm8rrT5ZFW3nj_iLYQNDA?usp=sharing) and place them in the corresponding directories in the project. **Please note that** if you are using an emulator or another rooted device, instead of a Google Pixel 5 device, it is necessary to **download the appropriate version of Frida**. Otherwise, the project will not launch successfully.

## Usage
1. Connect your rooted Google Pixel 5 mobile device to your local machine and enable USB debugging.
2. Run the Explorer module to monitor and record app activity and sensitive API access.
   ```sh
   cd Explorer
   python explorer.py
   ```
3. Use the recorded data as input for the Fidelity Checker module to evaluate whether app activity violates the Data Minimization Principle.
   ```sh
   cd FidelityChecker
   python fidelity_checker.py
   ```
4. Analyze the output report generated by the Fidelity Checker module to identify violations of the Data Minimization Principle.

## License
This project is licensed under the MIT License

## Contact
If you have any questions or suggestions, please feel free to contact us.

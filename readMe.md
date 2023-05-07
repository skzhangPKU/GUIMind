## GUIMind
---
A novel automated tool to detect violations of the Data Minimization Principle. The tool consists of two modules: Explorer and Fidelity Checker.


## Explorer
---
Explorer utilizes a reinforcement learning model to explore app activities and monitor access to sensitive APIs that require sensitive permissions.


## Fidelity Checker
---
Fidelity Checker uses the existing tool [APICOG](https://ieeexplore.ieee.org/abstract/document/9251054) to check whether an activity collects more sensitive permissions than the users expect.


Pretrained model, Frida (Google Pixel 5), and Standford NLP Parser:
https://drive.google.com/drive/folders/1FrDyUGGBHNazm8rrT5ZFW3nj_iLYQNDA?usp=sharing

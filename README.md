CARDIOWISE: Smart Home-Based Heart Monitoring System

(https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A privacy-preserving cardiac health monitoring system using federated learning on Raspberry Pi edge devices. This repository contains the source code, documentation, and implementation details for the CARDIOWISE project.

Table of Contents

-(#about-the-project)

    Key Features
    -(#system-architecture)
    -(#technology-stack)
    -(#getting-started)

        Prerequisites
        -(#hardware-setup)
        -(#software-installation)
        -(#how-to-run)
        -(#project-roadmap)

    Contributing

    License

    Acknowledgement

About The Project

Cardiovascular diseases are a leading cause of death globally, often due to a lack of continuous monitoring and timely diagnosis. While digital health solutions offer a path forward, they typically rely on centralized systems that require patients to upload sensitive health data to the cloud, creating significant privacy and security risks.  

The CARDIOWISE project addresses this challenge by developing a smart, decentralized, and privacy-first heart monitoring system. It leverages the power of Federated Learning (FL) and Edge Computing to perform real-time cardiac risk prediction directly within a user's home environment.  

The core principle is simple but powerful: raw patient data, collected from IoT sensors, never leaves the local edge device (a Raspberry Pi). Instead of data, only encrypted and anonymized machine learning model updates are shared with a central coordination server. This allows a global prediction model to be collaboratively trained and improved over time without ever compromising individual patient privacy.  

Key Features

This project is built on several novel concepts to create a trustworthy and effective healthcare solution :  

    🔒 Privacy-First by Design: All data preprocessing and model training occurs on-device. No raw physiological data is ever transmitted outside the user's home, ensuring full patient control over their health information.

    🤖 Federated Learning: The system uses the Flower framework to collaboratively train a global Long Short-Term Memory (LSTM) model across a network of distributed devices without centralizing the data.   

💡 Confidence-Aware Predictions: Unlike binary risk scores, CARDIOWISE quantifies the uncertainty of its predictions, providing a confidence level with each alert to help clinicians make more informed decisions.  

⚖️ Fair and Inclusive AI: The framework incorporates fairness-aware machine learning techniques to minimize performance disparities across different patient demographics (e.g., age, gender), promoting more equitable healthcare.  

⚡ Real-Time Edge AI: Lightweight LSTM models are deployed on Raspberry Pi nodes for continuous, real-time analysis of physiological signals like ECG, blood pressure, and SpO2, enabling early detection of cardiac anomalies.  

System Architecture

The CARDIOWISE system is designed with a three-layer architecture to ensure privacy, scalability, and real-time performance :  

    🏠 Home IoT Layer (Data Collection): At the bottom layer, wearable and home-based medical sensors (e.g., ECG monitors, blood pressure cuffs, pulse oximeters) continuously collect physiological data from the patient in their daily environment.

    엣 Edge Layer (Local Training & Inference): In the middle layer, a Raspberry Pi in each home acts as a secure edge node. This device is responsible for data preprocessing, local training of the machine learning model on the user's data, and real-time inference to detect potential health risks.

    ☁️ Coordination Layer (Federated Aggregation): At the top layer, a central server orchestrates the federated learning process. It receives encrypted model updates from all participating edge nodes and uses an aggregation strategy like Federated Averaging (FedAvg) to build an improved, more robust global model, which is then sent back to the clients.   

Technology Stack

    Hardware:

        Raspberry Pi 4 Cluster (Client Nodes & Server)   

AD8232 ECG Sensor  

UART-based Blood Pressure (BP) Sensor  

MAX30102 Heart Rate/SpO2 Sensor  

ADS1115 Analog-to-Digital Converter (ADC)  

Frameworks & Libraries:

    Flower (flwr)   

(https://pytorch.org/)  

scikit-learn  

Pandas & NumPy  

    Languages & Tools:

        Python

        Git & GitHub

        Docker (for future deployment)

Getting Started

To get a local copy up and running, follow these simple steps.

Prerequisites

Ensure you have Python 3.9+ and pip installed on your system.

    pythonsh
    python --version

    pip
    Bash

    pip --version

Hardware Setup

    Assemble your Raspberry Pi cluster hardware. A stackable case is recommended for organization and cooling.   

Connect all Raspberry Pi nodes and the server machine to a network switch using Ethernet cables.  

Connect the physiological sensors (ECG, BP, SpO2) to the GPIO pins of each Raspberry Pi client node, using an ADC like the ADS1115 for analog sensors.  

Software Installation

    Clone the repository to your server machine and each Raspberry Pi client.
    Bash

git clone [https://github.com/your_username/cardiowise.git](https://github.com/your_username/cardiowise.git)
cd cardiowise

It is highly recommended to create and activate a Python virtual environment on each machine.
Bash

python3 -m venv venv
source venv/bin/activate

Install the required Python packages.
Bash

    pip install -r requirements.txt

    (Note: You will need to create a requirements.txt file containing libraries like flwr, torch, numpy, pyserial, etc.)

How to Run

    Start the Federated Server:
    On your central server machine, run the server application. This will start the SuperLink which waits for clients to connect.
    Bash

python server/server.py

Start the Clients:
On each Raspberry Pi client node, run the client application. This script will connect to the server, download the model, train it on local (simulated or real) sensor data, and send the updates back.
Bash

    python client/client.py

    Observe the Process:
    Monitor the terminal output on both the server and client machines to see the federated learning rounds in action. The server will log received updates and aggregation events, while clients will log their local training progress.

Project Roadmap

This project follows a 3-year development plan as outlined in the research proposal :  

    Phase 1 (Year 1): Implement the baseline federated learning framework with a standard LSTM model and validate it on the hardware testbed.

    Phase 2 (Year 2): Integrate uncertainty quantification (confidence-awareness) and fairness-aware algorithms into the model and training process.

    Phase 3 (Year 3): Explore advanced FL techniques (e.g., fair Reinforcement Learning), conduct robustness testing, and develop a proof-of-concept visualization dashboard.

See the open issues for a full list of proposed features (and known issues).

Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

    Fork the Project

    Create your Feature Branch (git checkout -b feature/AmazingFeature)

    Commit your Changes (git commit -m 'Add some AmazingFeature')

    Push to the Branch (git push origin feature/AmazingFeature)

    Open a Pull Request

License

Distributed under the MIT License. See LICENSE.txt for more information.

Acknowledgement

This project is based on the research proposal "Smart Home-Based Heart Monitoring System Using Machine Learning" by Dr. Sadananda Behera (PI) and Dr. Upendra Kumar Sahoo (Co-PI) at the Department of Electronics and Communication Engineering, National Institute of Technology, Rourkela. 

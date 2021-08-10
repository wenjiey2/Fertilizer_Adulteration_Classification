# Fertilizer Adulteration Classification with Deep Learning

### Research Team

The research is led by Professor Hope Michelson ([hopecm@illinois.edu](hopecm@illinois.edu)) at the Department of Agricultural and Consumer Economics (ACE) at the University of Illinois at Urbana-Champaign (UIUC). Since I will be no longer in charge of this project, if you are interested in the research or the latest update for this APP, please reach out to aceuiucfertilizer@gmail.com.

### Motivation

The innovation aims to assist farmers in developing countries in detecting adulterated fertilizers. Our research group investigated fertilizer usages in Tanzania, and learned that farmers were reluctant to utilize fertilizers because they were concerned about the quality of fertilizers. Nevertheless, our study showed that most of the fertilizers were above legal standard, which aligned with the findings of prior literature. According to our survey, many farmers falsely associated clumped and discolored fertilizers with low quality. Such misconceptions led to a decreasing willingness to use fertilizers and a low level of agricultural productivity. Furthermore, it was difficult to differentiate pure fertilizers from adulterated ones with naked eyes, even for experienced government agents. As suspicion of fertilizer quality contributed significantly to its underusage, we sought to develop a mobile application to provide instant predictions of whether the fertilizer was adulterated.

### Project Overview

In order to fulfill the goals of this research, the team needed an Android APP which can help farmers from Tanzania identify whether the fertilizer they use is adulterated based on the fertilizer images they take. You can get started with installation of the APP [here](https://github.com/wenjiey2/Fertilizer_Adulteration_Detection/tree/main/app#installation) and the DNN model [here](https://github.com/wenjiey2/Fertilizer_Adulteration_Detection/tree/main/model/model_2.2#model-22).

### My Work

When I took over the project, I was provided the skeleton of the app and a very simple neural network as the classification model. In terms of the APP, I was able to provide a fully functional APP that supports user registration/login, camera usage/photo upload for classification and Firebase backup. On the model side, I was able to completely redesign the DNN, fine tune it and explore optimizations of the model such as pruning and quantization for deployment.

# Phishing Boat ⛵️🎣, 
## Phishing Website Detector
*Dawid Okwieka, Scott McNally*
*Email 1 : okwieka@kth.se*
*Email 2 : samcn@kth.se*


### Background
Phishing emails have become a bigger and more serious threat due to the increasing growth in digital communications using channels like email or SMS allowing cyber criminals to trick individuals and steal their personal data. Phishing also creates significant threats for businesses and led to $1.7 billion of losses for businesses and organizations in 2019 [1]. The improving language capabilities of large language models like ChatGPT and more sophisticated methods allow criminals to create more realistic and undistinguishable emails and messages and work on a larger scale targeting more individuals at once. We have researched existing approaches like detecting phishing and spam emails comparing the use of transformer models and LLM [3] as well as existing plugins focused on bank phishing.

### Problems and Goal
The problem we are hoping to tackle with this project is detection of phishing websites/emails before a user can even interact with them. This helps the user to not fall for phishing scams through typical behavioral hacking techniques like instinct, habits and time pressure. 
Through training a model on both verified valid websites and scam websites known to be used by phishing actors we hope to create a program that will be able to analyze html code and predict whether it has similar patterns to known phishing attacks.
We found labeled data sets that we plan to train our model on. The most prevalent being released by the University of Moratuwa. This contains a labeled set of 50,000 legitimate websites and 30,000 phishing websites. [2]
We plan to use Python to develop this model. We both have experience using python for different projects. It also seems to be the industry standard for training AI models like this and allows us to easily parse our dataset.

### Contributions
By developing this program we hope to create a program that can be used by businesses to secure themselves against internal phishing attacks drastically reducing the losses and reputational harm caused to them. In addition we would also be able to create a feeling of secureness during personal use.

### Milestone Chart
Weeks
Milestone
Week 1
Curating a dataset and project proposal
Week 2
Researching other solutions and planning our approach
Week 3
Begin implementing model
Week 4
Training our own model
Week 5
Continue training while adjusting parameters
Week 6
Christmas Holiday and continue any training that needs to be done
Week 7
Make inference / the code presentable
Week 8
Finish our report, framework through the project

### References
[1] How can phishing affect a business, 2023. Link: https://www.cybsafe.com/blog/how-can-phishing-affect-a-business/ (accessed on 11.11.2024)
[2] Phishing Websites Dataset, 2021. Link: https://data.mendeley.com/datasets/n96ncsr5g4/1 (accessed on 11.11.2024)
[3] Suhaima Jamal, Hayden Wimmer, Iqbal H. Sarker:  An improved transformer-based model for detecting phishing, spam and ham emails: A large language model approach, 2024. Link: https://onlinelibrary.wiley.com/doi/full/10.1002/spy2.402 (accessed on 11.11.2024)

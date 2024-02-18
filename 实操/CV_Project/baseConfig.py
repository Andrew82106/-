import os
import sys


rootFileName = 'CV_Project'
rootPath = __file__[:str(__file__).find(rootFileName) + len(rootFileName)]
CNNPath = os.path.join(rootPath, "CNN")
archivePath = os.path.join(rootPath, "archive")
archiveTrainingPath = os.path.join(archivePath, "Training")
archiveTrainingFemalePath = os.path.join(archiveTrainingPath, "female")
archiveTrainingMalePath = os.path.join(archiveTrainingPath, "male")
archiveValidationPath = os.path.join(archivePath, "Validation")
archiveValidationFemalePath = os.path.join(archiveValidationPath, "female")
archiveValidationMalePath = os.path.join(archiveValidationPath, "male")

sys.path.append(archivePath)
sys.path.append(rootPath)
sys.path.append(CNNPath)

algoLst = [CNNPath]

print(f"Project Loaded::\nrootPath:{rootPath}")
print(f"CNNPath:{CNNPath}")
print(f"archivePath:{archivePath}")
print(f"archiveTrainingPath:{archiveTrainingPath}")
print(f"archiveTrainingFemalePath:{archiveTrainingFemalePath}")
print(f"archiveTrainingMalePath:{archiveTrainingMalePath}")
print(f"archiveValidationPath:{archiveValidationPath}")
print(f"archiveValidationFemalePath:{archiveValidationFemalePath}")
print(f"archiveValidationMalePath:{archiveValidationMalePath}")
print(f"available algorithm:")
print("-"*100)
for Route in algoLst:
    print(f"-{Route}")
print("-"*100)
import sys
import yaml
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit, QHBoxLayout

class YamlReader(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('YAML File Reader')
        self.setGeometry(100, 100, 600, 400)

        mainLayout = QVBoxLayout()
        
        # Create horizontal layout for the log terminals
        yamlLayout = QVBoxLayout()

        self.textEdit = QTextEdit(self)
        self.textEdit.setReadOnly(True)
        yamlLayout.addWidget(self.textEdit)
        
        btnLoad = QPushButton('Load YAML File', self)
        btnLoad.clicked.connect(self.openFileNameDialog)
        yamlLayout.addWidget(btnLoad)
        
        mainLayout.addLayout(yamlLayout)
        
        trainLayout = QVBoxLayout()

        
        self.logTerminal = QTextEdit(self)
        self.logTerminal.setReadOnly(True)
        trainLayout.addWidget(self.logTerminal)
        

        btnTrain = QPushButton('train', self)
        trainLayout.addWidget(btnTrain)

        mainLayout.addLayout(trainLayout)


        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", 
                                                  "YAML Files (*.yaml *.yml);;All Files (*)", options=options)
        if fileName:
            self.loadYamlFile(fileName)

    def loadYamlFile(self, filePath):
        with open(filePath, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            self.textEdit.setText(str(data))
def main():
    app = QApplication(sys.argv)
    ex = YamlReader()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

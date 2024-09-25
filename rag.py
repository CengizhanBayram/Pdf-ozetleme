import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QHBoxLayout,
    QTabWidget, QListWidget, QListWidgetItem, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# OpenAI API anahtarını ayarla
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Lütfen OpenAI API anahtarınızı ayarlayın.")

class PDFProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.parent = parent

    def run(self):
        try:
            # PDF'den belgeleri yükle
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()

            # Metni parçalara ayır
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            texts = text_splitter.split_documents(documents)
            total_docs = len(texts)

            # Embedding modeli tanımlama
            embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # FAISS vektör mağazasını oluşturma
            self.parent.vector_store = FAISS.from_documents(texts, embedding)

            self.parent.retriever = self.parent.vector_store.as_retriever()

            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))

class AnswerThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, question, parent=None):
        super().__init__(parent)
        self.question = question
        self.parent = parent

    def run(self):
        try:
            if self.parent.retriever:
                # İlgili belgeleri al
                docs = self.parent.retriever.get_relevant_documents(self.question)

                # İçeriği birleştir
                context = "\n\n".join([doc.page_content for doc in docs])

                # Prompt oluşturma
                prompt = f"Context:\n{context}\n\nQuestion:\n{self.question}"
            else:
                # PDF yüklenmemişse, doğrudan kullanıcı sorusunu kullan
                prompt = self.question

            # LLM'den yanıt alma
            response = self.parent.llm.invoke(prompt)

            # Yanıtın içeriğini al
            self.finished.emit(response.content)
        except Exception as e:
            # Hata mesajını yanıt olarak gönder
            self.finished.emit(f"Üzgünüm, bir hata oluştu:\n{str(e)}")

class ChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF özetleyici")
        self.resize(800, 600)
        self.setWindowIcon(QIcon("icon.png"))  # Uygulama ikonu
        self.init_ui()
        self.documents = None
        self.vector_store = None
        self.retriever = None
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Logo ekleme
        logo = QLabel()
        pixmap = QPixmap("logo.png")  # Logo dosyasının yolu
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(200, 100, Qt.KeepAspectRatio))
        else:
            logo.setText("Logo Yüklenemedi")
        logo.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(logo)

        # Sekmeler
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # PDF Yükleme Sekmesi
        self.pdf_tab = QWidget()
        self.pdf_layout = QVBoxLayout()
        self.pdf_tab.setLayout(self.pdf_layout)

        self.upload_button = QPushButton("PDF Dosyasını Yükle")
        self.upload_button.setIcon(QIcon("upload_icon.png"))
        self.upload_button.clicked.connect(self.load_pdf)
        self.pdf_layout.addWidget(self.upload_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.pdf_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Lütfen bir PDF dosyası yükleyin veya doğrudan sohbet edin.")
        self.pdf_layout.addWidget(self.status_label)

        self.tabs.addTab(self.pdf_tab, "PDF Yükle")

        # Sohbet Sekmesi
        self.chat_tab = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_tab.setLayout(self.chat_layout)

        self.chat_history = QListWidget()
        self.chat_layout.addWidget(self.chat_history)

        message_layout = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Sorunuz:")
        message_layout.addWidget(self.question_input)

        self.ask_button = QPushButton()
        self.ask_button.setIcon(QIcon("send_icon.png"))
        self.ask_button.clicked.connect(self.ask_question)
        self.ask_button.setEnabled(True)  # PDF yüklemeden de soru sorabilmek için etkinleştirildi
        message_layout.addWidget(self.ask_button)

        self.chat_layout.addLayout(message_layout)

        self.tabs.addTab(self.chat_tab, "Sohbet")

        self.setLayout(main_layout)

    def load_pdf(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "PDF Dosyasını Seçin", "", "PDF Files (*.pdf)", options=options
        )
        if file_name:
            self.status_label.setText("PDF dosyası yükleniyor...")
            self.progress_bar.setValue(0)
            self.upload_button.setEnabled(False)
            self.thread = PDFProcessingThread(file_name, parent=self)
            self.thread.progress.connect(self.progress_bar.setValue)
            self.thread.finished.connect(self.pdf_loaded)
            self.thread.start()

    def pdf_loaded(self, success, error_message):
        self.upload_button.setEnabled(True)
        if success:
            self.status_label.setText("PDF dosyası yüklendi. Sorularınız artık PDF içeriğine göre yanıtlanacak.")
            QMessageBox.information(self, "Başarılı", "PDF dosyası başarıyla yüklendi.")
            self.tabs.setCurrentWidget(self.chat_tab)
        else:
            self.status_label.setText("PDF yükleme sırasında bir hata oluştu.")
            QMessageBox.critical(self, "Hata", f"PDF dosyası yüklenirken bir hata oluştu:\n{error_message}")

    def add_message(self, sender, message):
        item = QListWidgetItem()
        if sender == "user":
            item.setText(f"👤 Siz: {message}")
            item.setTextAlignment(Qt.AlignRight)
        else:
            item.setText(f"🤖 Bot: {message}")
            item.setTextAlignment(Qt.AlignLeft)
        self.chat_history.addItem(item)
        self.chat_history.scrollToBottom()

    def ask_question(self):
        user_question = self.question_input.text()
        if user_question.strip() == "":
            return

        self.add_message("user", user_question)
        self.question_input.clear()
        self.ask_button.setEnabled(False)
        self.status_label.setText("Yanıt alınıyor...")

        self.thread = AnswerThread(user_question, self)
        self.thread.finished.connect(self.display_answer)
        self.thread.start()

    def display_answer(self, response):
        self.add_message("bot", response)
        self.ask_button.setEnabled(True)
        self.status_label.setText("Sorunuzu sorabilirsiniz.")

def main():
    app = QApplication(sys.argv)

    # Uygulama stili
    app.setStyle("Fusion")
    # QSS ile stil ekleme
    style_sheet = """
    QWidget {
        font-size: 14px;
    }
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QLineEdit {
        padding: 5px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    QListWidget {
        background-color: #f0f0f0;
        border: none;
    }
    QLabel {
        font-weight: bold;
    }
    QProgressBar {
        height: 15px;
    }
    """
    app.setStyleSheet(style_sheet)

    chatbot = ChatbotApp()
    chatbot.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

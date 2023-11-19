import streamlit  as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css,bot_template,user_template
from langchain.llms import HuggingFaceHub 




def get_pdf_text(pdf_docs):
   text=""
   for pdf in pdf_docs:
      pdf_reader=PdfReader(pdf)
      for page in pdf_reader.pages:
         page.extract_text()
   return text 

def get_text_chunks(text):
   text_splitter = CharacterTextSplitter(
       separator="\n",
       chnks_size=1000,
       chunks_overlap=200,
       length_function=len
   )
   chunks=text_splitter.split_text(get_text_chunks)
   return chunks 


def get_vectorstore(text_chunks):
   #embedding=OpenAIEmbeddings()
   embeddings=HuggingFaceInstructEmbeddings(model_name="HuggingFaceH4/zephyr-7b-beta")
   vectorestore=FAISS.from_texts(text=text_chunks,embedding=embeddings)
   return vectorestore
def OpenAi():
   return 'hai'

def get_conversation_chain(vectorstore):
   llm=OpenAi()
   llm=HuggingFaceHub(repo_id="",model_kwarges={"temperature":0.5,"max_length":5})
   memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
   conversation_chain=ConversationalRetrievalChain.from_llm(
      llm=llm,
      retriever=vectorstore.ass_retriver(),
      memory=memory

   )
   return conversation_chain

def handle_userinput(user_question):
   reponse=st.session_state.conversation({'question':user_question})
   st.session_state.chat_history= reponse['chat_history']


   for i, message in enumerate(st.session_state.chat_history):
      if i % 2 == 0:
         st.write(user_template.replace("{{msg}}",message.content),unsafe_allow_html=True)
      else:
         st.write(bot_template.replace("{{msg}}",message.content),unsafe_allow_html=True)



def main(): 
   load_dotenv()
   st.set_page_config(page_title="chat with multiple PDFs")

   st.write(css,unsafe_allow_html=True)


   if"conversation" not in st.session_state:
      st.session_state.conversation=None
   if  "chat_history" not in st.session_state:
        st.session_state.chat_history=None

   st.header("chat with multiple PDFs:books")
   user_question=st.text_input("ask aquestion about your documents:")
   if user_question:
       handle_userinput(user_question)
  
   st.write(user_template.replace("{{msg}}","hello robot"),unsafe_allow_html=True)
   st.write(bot_template.replace("{{msg}}","hello human"),unsafe_allow_html=True)
   
   
   with st.sidebar:
      st.subheader("your document")
      PDFs_docs=st.file_uploader(
         "upload your PDFs here and click on 'proces'",accept_multiple_files=True)
      if st.button("process"):
         with st.spinner("processing"):
            #get pdf text
            raw_text=get_pdf_text(PDFs_docs)
            st.write(raw_text)
      

         #get  the text chuks
         text_chunks=get_text_chunks(raw_text)
         st.write(text_chunks)
      

        #creat vectore stores
         vectorstore=get_vectorstore(text_chunks)
        
        #creat conversation chain 
         st.session_state.conversation = get_conversation_chain(vectorstore)
         st.session_state.conversation
      

   


if __name__=='__main__':
    main() 
                                              




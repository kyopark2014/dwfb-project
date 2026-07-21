import streamlit as st 
import streamlit_paste_button as spb
import chat
import json
import os
import io
import mcp_config 
import asyncio
import logging
import sys
import strands_agent
import uuid
import utils
import skill
from notification_queue import NotificationQueue

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

config = utils.load_config()

# title
st.set_page_config(page_title='Agent', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],    
    "Agent": [
        "Strands Agent SDK를 활용한 Agent를 이용합니다."
    ],
    "이미지 분석": [
        "이미지를 선택하여 멀티모달을 이용하여 분석합니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Stands Agent SDK와 Agent Skills를 이용하여 효과적인 Agent를 구현합니다." 
        "상세한 코드는 [Github](https://github.com/kyopark2014/sam-project)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    options = [
        "일상적인 대화", 
        'RAG', 
        'Agent',         
        '이미지 분석'
    ] 
    mode = st.radio(label="원하는 대화 형태를 선택하세요. ", options=options, index=2)   
    st.info(mode_descriptions[mode][0])    

    strands_tools = ["current_time", "file_read", "file_write", "http_request"] 
    default_strands_tool_selections = ["current_time", "file_read", "file_write"]    
    
    # mcp selection    
    mcp_tools = [
        "tavily", 
        "RAG", 
        "aws documentation", 
        "trade_info", 
        "web_fetch",
        "websearch",
        "korea_weather",
        "사용자 설정"
    ]

    mcp_selections = {}
    # Default: prevent strands_selections undefined when not in Agent mode
    default_strands_tool_selections = config.get("default_strands_tool_selections") or default_strands_tool_selections
    strands_selections = {tool: tool in default_strands_tool_selections for tool in strands_tools}

    if mode=="Agent" or mode=="Agent (Chat)":
        # Skill Config JSON input
        st.subheader("⚙️ Skill Config")

        skill_selections = {}
        default_skill_selections, default_mcp_selections = utils.get_initial_tool_defaults()
        logger.info(f"default_skill_selections: {default_skill_selections}")
        with st.expander("Skill 옵션 선택", expanded=True):
            available_skill_info = skill.available_skill_info("base")
            for s in available_skill_info:
                default_value = s["name"] in default_skill_selections
                skill_selections[s["name"]] = st.checkbox(s["name"], key=f"skill_{s['name']}", value=default_value, help=s["description"], disabled=False)
    
        selected_skills = [name for name, is_selected in skill_selections.items() if is_selected]
        logger.info(f"selected_skills: {selected_skills}")

        # Strands Tool Config JSON input
        st.subheader("⚙️ Strands Tool Config")
        
        strands_tool_selections = {}
        default_strands_tool_selections = config.get("default_strands_tool_selections") or default_strands_tool_selections
        logger.info(f"default_strands_tool_selections: {default_strands_tool_selections}")
        
        with st.expander("Strands Tool 옵션 선택", expanded=True):
            for tool in strands_tools:
                default_value = tool in default_strands_tool_selections
                strands_tool_selections[tool] = st.checkbox(tool, key=f"strands_tool_{tool}", value=default_value, disabled=False)
        
        selected_strands_tools = [name for name, is_selected in strands_tool_selections.items() if is_selected]
        logger.info(f"selected_strands_tools: {selected_strands_tools}")
        strands_selections = strands_tool_selections  # used at line 377

        if selected_strands_tools != config.get("default_strands_tool_selections"):
            config["default_strands_tool_selections"] = selected_strands_tools
            with open(utils.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logger.info("save to config.json")

        # MCP Config JSON input
        st.subheader("⚙️ MCP Config")

        with st.expander("MCP 옵션 선택", expanded=True):
            for option in mcp_tools:
                default_value = option in default_mcp_selections
                mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
                
        if mcp_selections["사용자 설정"]:
            mcp = {}
            try:
                with open("user_defined_mcp.json", "r", encoding="utf-8") as f:
                    mcp = json.load(f)
                    logger.info(f"loaded user defined mcp: {mcp}")
            except FileNotFoundError:
                logger.info("user_defined_mcp.json not found")
                pass
            
            mcp_json_str = json.dumps(mcp, ensure_ascii=False, indent=2) if mcp else ""
            
            mcp_info = st.text_area(
                "MCP 설정을 JSON 형식으로 입력하세요",
                value=mcp_json_str,
                height=150
            )
            logger.info(f"mcp_info: {mcp_info}")

            if mcp_info:
                try:
                    mcp_config.mcp_user_config = json.loads(mcp_info)
                    logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")                    
                    st.success("JSON 설정이 성공적으로 로드되었습니다.")                    
                except json.JSONDecodeError as e:
                    st.error(f"JSON 파싱 오류: {str(e)}")
                    st.error("올바른 JSON 형식으로 입력해주세요.")
                    logger.error(f"JSON 파싱 오류: {str(e)}")
                    mcp_config.mcp_user_config = {}
            else:
                mcp_config.mcp_user_config = {}
                
            with open("user_defined_mcp.json", "w", encoding="utf-8") as f:
                json.dump(mcp_config.mcp_user_config, f, ensure_ascii=False, indent=4)
            logger.info("save to user_defined_mcp.json")
        
        mcp_servers = [server for server, is_selected in mcp_selections.items() if is_selected]
        if (
            selected_skills != default_skill_selections
            or mcp_servers != default_mcp_selections
        ):
            utils.save_favorite_tools(skills=selected_skills, mcp_servers=mcp_servers)


    else:
        mcp_servers = []
        selected_skills = []


    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        (
            "Claude 4.6 Sonnet",
            "Claude 4.7 Opus",
            "Claude 4.6 Opus",
            "Claude 4.5 Haiku",
            "Claude 4.5 Sonnet",
            "Claude 4.5 Opus",  
            "OpenAI GPT 5.4",
            "OpenAI GPT 5.5",
            "OpenAI GPT 5.6 Sol",
            "OpenAI GPT 5.6 Terra",
            "OpenAI GPT 5.6 Luna",
            "OpenAI OSS 120B",
            "OpenAI OSS 20B",
            "Nova 2 Lite",
            "Nova Premier", 
            "Nova Pro", 
            "Nova Lite", 
            "Nova Micro",       
        ), index=0
    )

    # skill checkbox
    select_skillMode = st.checkbox('Skill Mode', value=True)
    skillMode = 'Enable' if select_skillMode else 'Disable'    

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    
    # extended thinking of claude 3.7 sonnet
    reasoningMode = 'Disable'
    if modelName == 'Claude 3.7 Sonnet' or modelName == 'Claude 4 Sonnet' or modelName == 'Claude 4 Opus':
        select_reasoning = st.checkbox('Reasoning', value=False)
        reasoningMode = 'Enable' if select_reasoning else 'Disable'
        logger.info(f"reasoningMode: {reasoningMode}")

    uploaded_file = None
    pasted_image = None

    def safe_paste_button(label, key):
        """streamlit-paste-button 래퍼: 내부 이미지 디코딩 실패 시 안전하게 처리"""
        try:
            result = spb.paste_image_button(label, key=key, errors="ignore")
            if result.image_data is not None:
                return result.image_data
        except Exception as e:
            logger.warning(f"clipboard paste error: {e}")
        return None

    if mode == '이미지 분석':
        st.subheader("🌇 이미지 업로드")
        uploaded_file = st.file_uploader("이미지 분석을 위한 파일을 선택합니다.", type=["png", "jpg", "jpeg"], key=chat.fileId)

        st.markdown("**또는** 화면 캡처를 붙여넣으세요:")
        pasted_image = safe_paste_button("📋 클립보드에서 붙여넣기", key="paste_image")

    elif mode=="RAG" or mode=="Agent" or mode=="Agent (Chat)":
        st.subheader("📋 문서 업로드")
        uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)
    
    selected_strands_tools = [tool for tool, is_selected in strands_selections.items() if is_selected]
    selected_mcp_servers = [server for server, is_selected in mcp_selections.items() if is_selected]
    
    chat.update(modelName, reasoningMode, debugMode, skillMode)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")

st.title('🔮 '+ mode)  

if clear_button==True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages():
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:
                for url in message["images"]:
                    logger.info(f"url: {url}")
                    # Only process image URLs or image files; skip non-images like .md, .txt
                    is_http = url.startswith("http://") or url.startswith("https://")
                    image_ext = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".ico")
                    if not (is_http or any(url.lower().endswith(ext) for ext in image_ext)):
                        continue
                    file_name = url[url.rfind("/") + 1:] if "/" in url else url
                    try:
                        st.image(url, caption=file_name, use_container_width=True)
                    except Exception as e:
                        logger.warning(f"st.image failed for {url}: {e}")            

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []     
    uploaded_file = None   
    
    st.session_state.greetings = False
    
    chat.clear_chat_history()
    st.rerun()

file_name = ""
file_bytes = None

if pasted_image is not None and clear_button==False:
    buf = io.BytesIO()
    pasted_image.save(buf, format="PNG")
    file_bytes = buf.getvalue()

    if mode == '이미지 분석':
        st.image(pasted_image, caption="붙여넣은 이미지 미리보기", use_container_width=True)

if uploaded_file is not None and clear_button==False:
    logger.info(f"uploaded_file.name: {uploaded_file.name}")
    if uploaded_file.name:
        logger.info(f"csv type? {uploaded_file.name.lower().endswith(('.csv'))}")

    if uploaded_file and clear_button==False and mode == '이미지 분석':
        st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)
        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

    elif uploaded_file.name:
        chat.initiate()

        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            logger.info(f"status: {status}")
            st.info(status)

        file_name = uploaded_file.name
        logger.info(f"uploading... file_name: {file_name}")
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"file_url: {file_url}")

        utils.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        if debugMode=='Enable':
            logger.info(f"status: {status}")
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        logger.info(f"msg: {msg}")

        st.write(msg)

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")
    #logger.info(f"is_updated: {agent.is_updated}")

    with st.chat_message("assistant"):
        image_urls = []

        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)            
            response = st.write_stream(stream)
            logger.info(f"response: {response}")

            chat.save_chat_history(prompt, response)

        elif mode == 'RAG':            
            # knowlege base retrieval
            response = chat.run_rag_with_knowledge_base(prompt, st)          
            st.markdown(response)                 

            # retrieve and generate
            # containers = {
            #     "notification": [st.empty() for _ in range(1000)],
            #     "message": st.empty()
            # }
            # response = chat.run_rag_using_retrieve_and_generate(prompt, notification_queue)
                        
            logger.info(f"response: {response}")
            chat.save_chat_history(prompt, response)

        elif mode == '이미지 분석':
            if file_bytes is None:
                st.error("이미지를 먼저 업로드하거나 클립보드에서 붙여넣으세요.")
                st.stop()
            else:
                with st.status("thinking...", expanded=True, state="running") as status:
                    summary = chat.summarize_image(file_bytes, prompt, st)
                    st.write(summary)

                    artifacts_dir = strands_agent.ARTIFACTS_DIR
                    os.makedirs(artifacts_dir, exist_ok=True)
                    artifact_name = f"image_summary_{uuid.uuid4().hex}.md"
                    artifact_path = os.path.join(artifacts_dir, artifact_name)
                    md_body = summary if isinstance(summary, str) else str(summary)
                    with open(artifact_path, "w", encoding="utf-8") as f:
                        f.write(md_body)

                    artifact_url = chat.upload_to_s3(md_body.encode("utf-8"), artifact_name)
                    if artifact_url:
                        st.markdown(
                            f"마크다운 artifact가 저장되었습니다. "
                            f"[S3 링크]({artifact_url}) · 로컬: `{artifact_path}`"
                        )
                        assistant_content = (
                            f"{md_body}\n\n---\n\n"
                            f"[분석 요약 (markdown artifact)]({artifact_url})"
                        )
                    else:
                        st.warning(
                            f"S3 업로드에 실패했거나 버킷/공유 URL이 설정되지 않았습니다. "
                            f"로컬 artifact: `{artifact_path}`"
                        )
                        assistant_content = md_body

                    chat.save_chat_history("문서 분석 결과", assistant_content)

                    st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                
        elif mode == 'Agent':
            with st.status("thinking...", expanded=True, state="running") as status:
                notification_queue = NotificationQueue(container=status)

                skill_list = selected_skills if selected_skills else []
                logger.info(f"skill_list: {skill_list}")

                response, image_urls = asyncio.run(strands_agent.run_strands_agent(
                    query=prompt, 
                    strands_tools=selected_strands_tools, 
                    mcp_servers=selected_mcp_servers, 
                    skill_list=skill_list,
                    notification_queue=notification_queue))

        if chat.debug_mode == 'Disable':
           st.markdown(response)
        
        for url in image_urls:
            logger.info(f"url: {url}")
            # Only process image URLs or image files; skip non-images like .md, .txt
            is_http = url.startswith("http://") or url.startswith("https://")
            image_ext = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".ico")
            is_image = is_http or any(url.lower().endswith(ext) for ext in image_ext)
            if not is_image:
                continue
            file_name = url[url.rfind("/") + 1:] if "/" in url else url
            try:
                st.image(url, caption=file_name, use_container_width=True)
            except Exception as e:
                logger.warning(f"st.image failed for {url}: {e}")      

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": image_urls if image_urls else []
        })
    
    


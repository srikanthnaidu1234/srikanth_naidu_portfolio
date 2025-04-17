import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Srikanth Naidu | Portfolio",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1565C0;
        padding-bottom: 0.3rem;
    }
    .company {
        font-weight: 600;
        color: #1976D2;
    }
    .position {
        font-weight: 500;
        font-style: italic;
        color: #2196F3;
    }
    .date {
        font-size: 0.9rem;
        color: #616161;
    }
    .bullet-point {
        margin-left: 1rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.2rem;
        border-radius: 3px;
    }
    .project-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 5px solid #1976D2;
    }
    .contact-info {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .contact-icon {
        margin-right: 0.5rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .skill-category {
        font-weight: 600;
        color: #1565C0;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }
    .skill-item {
        background-color: #E3F2FD;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .progress-container {
        margin-bottom: 0.8rem;
    }
    .progress-label {
        font-weight: 500;
        margin-bottom: 0.2rem;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .education-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        border-left: 5px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Functions to calculate experience
def get_experience_years():
    # Calculate based on the resume data
    # Starting from first job in 2016 to current date
    start_date = datetime(2016, 9, 1)
    current_date = datetime.now()
    return round((current_date - start_date).days / 365, 1)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Profile image placeholder - replace with actual image if available
    st.image("https://via.placeholder.com/300", caption="Srikanth Naidu")
    
    st.markdown("<h3>Contact Information</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="contact-info">
        <span class="contact-icon">📧</span><span>srikanth.dbit04@gmail.com</span>
    </div>
    <div class="contact-info">
        <span class="contact-icon">📱</span><span>+1-201-780-9842</span>
    </div>
    <div class="contact-info">
        <span class="contact-icon">📍</span><span>New Jersey, USA</span>
    </div>
    <div class="contact-info">
        <span class="contact-icon">🔗</span><a href="https://linkedin.com/in/srikanth-naidu-aa6581170" target="_blank">LinkedIn</a>
    </div>
    <div class="contact-info">
        <span class="contact-icon">💻</span><a href="https://github.com/srikanthnaidu1234" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Skills visualization
    st.markdown("<h3>Skills Overview</h3>", unsafe_allow_html=True)
    
    # Skill categories and proficiency
    skills_data = {
        "Cloud & Data": 0.9,
        "Programming": 0.85,
        "ML & AI": 0.9,
        "Deep Learning": 0.8,
        "LLMs": 0.85,
        "MLOps": 0.75,
        "Distributed Computing": 0.8
    }
    
    for skill, proficiency in skills_data.items():
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-label">{skill}</div>
            <div style="height: 10px; background-color: #E0E0E0; border-radius: 5px;">
                <div style="height: 100%; width: {int(proficiency * 100)}%; background-color: #1976D2; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Srikanth Naidu</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="font-size: 1.2rem;">
    Data Science & AI Engineer with <b>""" + str(get_experience_years()) + """+ years</b> of cross-functional experience in 
    data analytics, data engineering, machine learning, and quality assurance.
</p>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Years of Experience", f"{get_experience_years()}")
with col2:
    st.metric("Projects Completed", "10+")
with col3:
    st.metric("Technical Skills", "40+")
with col4:
    st.metric("Industries", "Automotive, Telecom, Healthcare")

# Navigation tabs
tabs = st.tabs(["About", "Experience", "Education", "Projects", "Skills & Tools", "Contact"])

# About Tab
with tabs[0]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Professional Summary</h2>', unsafe_allow_html=True)
    st.markdown("""
    Versatile and results-driven Data Science & AI Engineer with extensive cross-functional experience spanning data analytics, 
    data engineering, machine learning, and quality assurance. Adept at building and optimizing scalable data pipelines, 
    deploying end-to-end AI solutions, and leveraging cloud platforms like Azure to drive business impact.
    
    I have a proven track record in applying advanced deep learning techniques including LLMs and reinforcement learning, 
    implementing MLOps best practices, and utilizing parallel computing across both academic and industry settings. 
    My passion lies in solving complex problems at scale and advancing intelligent automation for real-world applications.
    """)
    
    st.markdown('<h2 class="section-header">Core Competencies</h2>', unsafe_allow_html=True)
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("""
        - **Data Engineering & Analytics**
          - Big Data Processing
          - ETL Pipeline Development
          - Cloud Data Architecture
        - **Machine Learning & AI**
          - Deep Learning Models
          - Reinforcement Learning
          - LLM Implementation
        """)
    with comp_col2:
        st.markdown("""
        - **Cloud & MLOps**
          - Azure Data Platform
          - Model Deployment
          - CI/CD for ML Pipelines
        - **Technical Leadership**
          - Cross-functional Collaboration
          - Problem Solving
          - Performance Optimization
        """)
    
    # Timeline visualization
    st.markdown('<h2 class="section-header">Career Timeline</h2>', unsafe_allow_html=True)
    
    timeline_data = {
        "Position": [
            "Graduate Teaching Assistant", 
            "Data Analyst & Data Engineer", 
            "System Engineer", 
            "Senior Associate Test Engineer"
        ],
        "Company": [
            "NJIT", 
            "Mercedes-Benz R&D", 
            "TCS (Tech-Orbit)", 
            "NTT DATA"
        ],
        "Start": [
            "2024-08", 
            "2022-04", 
            "2021-09", 
            "2016-09"
        ],
        "End": [
            "2025-05", 
            "2024-01", 
            "2022-03", 
            "2020-08"
        ]
    }
    timeline_df = pd.DataFrame(timeline_data)
    
    # Convert dates for plotting
    timeline_df["Start_dt"] = pd.to_datetime(timeline_df["Start"])
    timeline_df["End_dt"] = pd.to_datetime(timeline_df["End"])
    timeline_df["Position_Company"] = timeline_df["Position"] + " at " + timeline_df["Company"]
    
    fig = px.timeline(
        timeline_df, 
        x_start="Start_dt", 
        x_end="End_dt", 
        y="Position_Company",
        color="Company",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"Position_Company": "Role"}
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        title="Professional Journey",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Experience Tab
with tabs[1]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Professional Experience</h2>', unsafe_allow_html=True)
    
    # Experience 1
    st.markdown("""
    <div class="company">New Jersey Institute of Technology</div>
    <div class="position">Graduate Teaching Assistant</div>
    <div class="date">August 2024 - May 2025 | New Jersey, US</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Teaching Assistant for course Data Analytics in R & Fundamentals and Principles of Data Science at NJIT
    - Support students in mastering core data science concepts, statistical methods, and practical applications using R and Python programming
    """)
    
    # Experience 2
    st.markdown("""
    <div class="company">Mercedes-Benz Research & Development</div>
    <div class="position">Data Analyst & Data Engineer</div>
    <div class="date">April 2022 - January 2024 | India</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Worked on scalable pipelines in Azure Data Factory (ADF) to ingest and process 10TB+ dataset through APIs, on-prem databases, Azure SQL DB and Data Lake Gen2
    - Enhanced data processing speed through PySpark using Azure Databricks for preprocessing, transformation, & aggregation
    - Delivered 145 actionable insights by implementing a real-time analytics solution displaying advanced data visualization capabilities using internal sensors for Mercedes development vehicle
    - Created Tableau dashboards for EU7 engine development to visualize performance trends analysis, enhancing monitoring capabilities & insights
    - Contributed to the deployment of scalable ML models for predictive analytics in collaboration with the engineering team
    """)
    
    # Experience 3
    st.markdown("""
    <div class="company">TCS (Tech-Orbit)</div>
    <div class="position">System Engineer</div>
    <div class="date">September 2020 - March 2021 | India</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Developed API monitoring, enabling real-time API-health check which in-turn assisted in fast-tracked fashion to identify and resolve 50+ critical API defects (e.g., authentication failures, data mismatches) during UAT, ensuring seamless integration with third-party systems
    - Integrated API tests by partnering with cross-functional teams (DevOps, developers) to reduce defect leakage in Agile releases
    - Designed test strategies and documented 150+ test cases for healthcare APIs, aligning with client requirements for eligibility checks, claims processing, and member data management
    """)
    
    # Experience 4
    st.markdown("""
    <div class="company">NTT DATA</div>
    <div class="position">Senior Associate Test Engineer</div>
    <div class="date">September 2016 - August 2020 | India & UK</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    - Led a 5-member QA team for BT Telecom projects, ensuring 100% on-time delivery for project releases through end-to-end testing for quality assurance in EE client in 2019
    - Successfully addressed 450+ critical defects, meeting strict deadlines for 15+ high-priority releases utilizing JIRA, Git, & Jenkins tools, resulting in improved release quality
    - Collaborated with DevOps & product teams to integrate automated tests into CI/CD workflows, leveraging Selenium, Jenkins, & Cucumber for enhanced testing efficiency resulting in a 60% reduction in pre-production defects
    - Resolved critical bottlenecks by conducting thorough root-cause analysis, promoting stakeholder engagement, & consistently achieving demanding project deadlines
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Education Tab
with tabs[2]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Academic Background</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="education-card">
            <h3>M.S. in Data Science</h3>
            <p><strong>New Jersey Institute of Technology</strong></p>
            <p><em>January 2024 - May 2025</em></p>
            <p><strong>Coursework:</strong></p>
            <ul>
                <li>Data Analytics with R</li>
                <li>Applied Statistics</li>
                <li>Machine Learning</li>
                <li>Deep Learning</li>
                <li>Artificial Intelligence</li>
                <li>Big Data</li>
                <li>Application in Parallel Computing</li>
                <li>Reinforcement Learning</li>
                <li>Machine Learning for Time Series Data and Forecasting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="education-card">
            <h3>M.Tech in Data Science</h3>
            <p><strong>Birla Institute of Technology (BITs)</strong></p>
            <p><em>March 2020 - March 2022</em></p>
            <p><strong>Focus Areas:</strong></p>
            <ul>
                <li>Advanced Data Analytics</li>
                <li>Machine Learning Algorithms</li>
                <li>Database Management Systems</li>
                <li>Big Data Technologies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Academic Projects</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="project-card">
        <h3>Hand Gesture Recognition for Assembly Tasks</h3>
        <p><em>M.S Project | August 2024 - December 2024</em></p>
        <ul>
            <li>Developed a deep learning-based classifier to distinguish between screwing and unscrewing tasks using real-time 3D hand pose data from the HoloAssist dataset</li>
            <li>Leveraged multi-modal sensor streams (RGB video, hand pose trajectories, and contextual annotations) to enhance task understanding in collaborative assembly/disassembly environments</li>
            <li>Research contributes to bridging human manipulation understanding with intelligent robotic systems for industrial automation</li>
        </ul>
    </div>
    
    <div class="project-card">
        <h3>Multi-Modal Deep Learning Framework for Stock Market Prediction</h3>
        <p><em>January 2025 - May 2025</em></p>
        <ul>
            <li>Designed a multi-modal deep learning architecture integrating numerical, textual, and sentiment data to predict stock price movements</li>
            <li>Implemented attention-based mechanisms for enhanced interpretability and real-time adaptability of the model</li>
        </ul>
    </div>
    
    <div class="project-card">
        <h3>Optimizing T5 Model (LLM) Training via 3D Parallelism in Mid Size Cluster</h3>
        <p><em>February 2025 - April 2025</em></p>
        <ul>
            <li>Conducted large-scale experiments on GPU clusters to evaluate the performance and scalability of T5 model training using tensor, data, model and pipeline parallelism</li>
            <li>Analyzed throughput, memory efficiency, and convergence rates across multiple parallel training configurations</li>
        </ul>
    </div>
    
    <div class="project-card">
        <h3>Reinforcement Learning for Cleaning Robot Navigation</h3>
        <p><em>March 2025 - May 2025</em></p>
        <ul>
            <li>Developed a reinforcement learning-based navigation policy using DDPG, PPO and actor-critic(SAC) algorithms for an indoor cleaning robot</li>
            <li>Simulated real-world cleaning scenarios using gym env to train and test navigation strategies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Projects Tab
with tabs[3]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Featured Projects</h2>', unsafe_allow_html=True)
    
    # Project filters
    project_categories = ["All", "Machine Learning", "Deep Learning", "Reinforcement Learning", "LLM", "Data Engineering"]
    selected_category = st.selectbox("Filter projects by category:", project_categories)
    
    # Project data
    projects = [
        {
            "title": "Hand Gesture Recognition System",
            "description": "Deep learning-based classifier to distinguish between screwing and unscrewing tasks using real-time 3D hand pose data from the HoloAssist dataset.",
            "technologies": ["PyTorch", "Computer Vision", "3D Pose Estimation"],
            "category": "Deep Learning",
            "image": "https://via.placeholder.com/300x200?text=Hand+Gesture+Recognition"
        },
        {
            "title": "Stock Market Prediction Framework",
            "description": "Multi-modal deep learning architecture integrating numerical, textual, and sentiment data to predict stock price movements with attention mechanisms.",
            "technologies": ["TensorFlow", "NLP", "Time Series Analysis"],
            "category": "Deep Learning",
            "image": "https://via.placeholder.com/300x200?text=Stock+Market+Prediction"
        },
        {
            "title": "T5 Model Training Optimization",
            "description": "Large-scale optimization of T5 model training using tensor, data, model and pipeline parallelism on GPU clusters.",
            "technologies": ["DeepSpeed", "Megatron-LM", "CUDA", "MPI"],
            "category": "LLM",
            "image": "https://via.placeholder.com/300x200?text=T5+Model+Optimization"
        },
        {
            "title": "Cleaning Robot Navigation",
            "description": "Reinforcement learning-based navigation policy for indoor cleaning robots using DDPG, PPO and SAC algorithms.",
            "technologies": ["Gym", "PyTorch", "Stable Baselines"],
            "category": "Reinforcement Learning",
            "image": "https://via.placeholder.com/300x200?text=Robot+Navigation"
        },
        {
            "title": "Mercedes Data Pipeline",
            "description": "Scalable data ingestion and processing pipeline in Azure Data Factory handling 10TB+ dataset for automobile analytics.",
            "technologies": ["Azure Data Factory", "PySpark", "Databricks", "Data Lake"],
            "category": "Data Engineering",
            "image": "https://via.placeholder.com/300x200?text=Data+Pipeline"
        },
        {
            "title": "Automated API Testing Framework",
            "description": "Real-time API monitoring system with automated testing capabilities for healthcare data integration.",
            "technologies": ["API Testing", "CI/CD", "Jenkins", "Python"],
            "category": "Data Engineering",
            "image": "https://via.placeholder.com/300x200?text=API+Testing"
        }
    ]
    
    # Filter projects based on selection
    if selected_category != "All":
        filtered_projects = [p for p in projects if p["category"] == selected_category]
    else:
        filtered_projects = projects
    
    # Display projects in grid
    cols = st.columns(2)
    for i, project in enumerate(filtered_projects):
        with cols[i % 2]:
            st.image(project["image"], use_column_width=True)
            st.markdown(f"### {project['title']}")
            st.markdown(f"**Category:** {project['category']}")
            st.markdown(project["description"])
            st.markdown(f"**Technologies:** {', '.join(project['technologies'])}")
            st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Skills & Tools Tab
with tabs[4]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Technical Expertise</h2>', unsafe_allow_html=True)
    
    # Group the skills
    skills = {
        "Related Knowledge": [
            "Neural network architectures", "Distributed training", "Mixed precision training", "Model parallelism", "Data parallelism", 
            "GPU/MPI acceleration", "Gradient accumulation", "Transformer", "Attention mechanisms", "Hyperparameter tuning", 
            "Supervised learning", "Unsupervised learning", "Feature engineering", "Cross-validation", "Ensemble methods", 
            "Gradient boosting", "Model evaluation metrics", "Decision trees", "Policy optimization", "Value functions", 
            "Multi-agent systems", "Environment simulation", "Reward shaping", "Markov Decision Processes", "Q-learning", 
            "Actor-critic methods", "ETL pipelines", "Data pipeline orchestration", "CI/CD automation", "Platform as a Service (PaaS)", 
            "Serverless computing", "Data lakehouse architecture", "Cloud resource management", "Fine-tuning", 
            "Parameter-efficient training", "Retrieval-Augmented Generation (RAG)", "Knowledge graphs", "Context window optimization", 
            "Quantization techniques", "Token optimization", "Inference acceleration", "LLM evaluation", 
            "Chain-of-thought prompting", "In-context learning", "Few-shot learning", "Parallel computing", "GPU programming", 
            "Multi-node computing", "Job scheduling", "Message passing", "Memory optimization", "Task parallelism", "Scalable algorithms"
        ],
        "Cloud Platforms": [
            "Azure Data Factory", "Databricks", "Data Lake", "Azure SQL Database", "Azure Functions", "Azure DevOps"
        ],
        "Languages": [
            "Python", "Java", "C", "SQL", "R"
        ],
        "Databases": [
            "MySQL", "PostgreSQL", "MongoDB", "Apache Cassandra", "Apache HBase", "Hive", "Pinecone", "Milvus"
        ],
        "Deep Learning": [
            "TensorFlow", "PyTorch", "DeepSpeed", "Megatron-LM"
        ],
        "Machine Learning": [
            "Scikit-learn", "XGBoost", "LightGBM", "Optuna", "Gridsearch"
        ],
        "Reinforcement Learning": [
            "Gym", "Stable Baselines", "pybullet"
        ],
        "LLM": [
            "Hugging Face Transformers", "Langgraph", "PEFT", "LoRA", "QLoRA", "LlamaIndex", "LangSmith", "DSPy", "LMQL", "Guidance", "vLLM", "GGML", "GPTQ"
        ],
        "MLOps": [
            "MLflow", "Weights & Biases"
        ],
        "Web Development": [
            "Flask", "Streamlit", "FastAPI", "Gradio", "Django"
        ],
        "DevOps": [
            "Docker", "Github"
        ],
        "Monitoring": [
            "Grafana", "Splunk", "Apigee", "TensorBoard"
        ],
        "Visualization": [
            "Tableau", "Power BI", "Plotly"
        ],
        "Distributed Systems": [
            "High Performance Computing (HPC)", "SLURM", "MPI", "CUDA Programming", "OpenMP", "OpenCL"
        ]
    }
    
    # Add a tab for each skill category
    skill_tabs = st.tabs(list(skills.keys()))
    
    for i, category in enumerate(skills.keys()):
        with skill_tabs[i]:
            st.markdown(f"<h3>{category}</h3>", unsafe_allow_html=True)
            
            # Create a word cloud-like display with skill bubbles
            html_skills = ""
            for skill in skills[category]:
                html_skills += f'<span class="skill-item">{skill}</span>'
            
            st.markdown(f"""
            <div style="line-height: 2.5;">
                {html_skills}
            </div>
            """, unsafe_allow_html=True)
    
    # Skills visualization
    st.markdown('<h2 class="section-header">Skills Distribution</h2>', unsafe_allow_html=True)
    
    # Sample data for visualization
    skill_distribution = {
        "Data Engineering": 25,
        "Machine Learning": 20,
        "Deep Learning": 15,
        "MLOps": 10,
        "LLM": 15,
        "Cloud Technologies": 10,
        "Distributed Computing": 5
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(skill_distribution.keys()),
        values=list(skill_distribution.values()),
        hole=.4,
        marker_colors=px.colors.qualitative.Safe
    )])
    
    fig.update_layout(
        title_text="Technical Skills Distribution",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Contact Tab
with tabs[5]:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Get In Touch</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Contact Information
        
        Feel free to reach out to me through any of the following channels:
        
        - 📧 **Email:** srikanth.dbit04@gmail.com
        - 📱 **Phone:** +1-201-780-9842
        - 📍 **Location:** New Jersey, USA
        - 🔗 **LinkedIn:** [linkedin.com/in/srikanth-naidu-aa6581170](https://linkedin.com/in/srikanth-naidu-aa6581170)
        - 💻 **GitHub:** [github.com/srikanthnaidu1234](https://github.com/srikanthnaidu1234)
        """)
    
    with col2:
        # Contact form
        st.markdown("### Send Me a Message")
        
        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            subject = st.text_input("Subject")
            message = st.text_area("Message")
            
            submitted = st.form_submit_button("Send Message")
            
            if submitted:
                st.success("Thank you for your message! I'll get back to you soon.")
    
    # Schedule a meeting
    st.markdown('<h2 class="section-header">Schedule a Meeting</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        If you'd like to discuss potential opportunities, collaborations, or projects, 
        please feel free to schedule a meeting with me.
        
        I'm available for:
        - Technical discussions
        - Project consultations
        - Job interviews
        - Academic collaborations
        """)
    
    with col2:
        meeting_date = st.date_input("Select a date")
        meeting_time = st.selectbox(
            "Select a time",
            ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]
        )
        meeting_type = st.radio(
            "Meeting type",
            ["Video Call", "Phone Call", "In-person (NJ area)"]
        )
        
        if st.button("Request Meeting"):
            st.success(f"Meeting request submitted for {meeting_date} at {meeting_time}. I'll confirm the details via email soon!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #e0e0e0;">
    <p>© 2025 Srikanth Naidu | Data Science & AI Portfolio</p>
    <p style="font-size: 0.8rem;">Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import xgboost
import shap
from sklearn.model_selection import train_test_split
from groq import Groq
import os
import random
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


app = Flask(__name__)

# Initialize Groq client
client = Groq(
    api_key= os.getenv("API_KEY")
    # os.getenv("API_KEY")

)

# Global variables to store model and data
model = None
X_test = None
feature_to_strategy_mapping = {
    'missed_payments_6months': [
        'Offer flexible payment plans',
        'Set up automated payment reminders',
        'Provide temporary payment relief'
    ],
    'monthly_bill_amount': [
        'Review and suggest cost-effective plans',
        'Offer bundled services for better value',
        'Provide personalized discount options'
    ],
    'payment_amount_trend': [
        'Conduct billing review consultation',
        'Suggest budget-friendly alternatives',
        'Offer loyalty rewards for consistent payments'
    ],
    'usage_trend_percent': [
        'Analyze usage patterns and suggest optimal plans',
        'Provide usage optimization tips',
        'Offer peak/off-peak usage benefits'
    ],
    'peak_hour_usage_ratio': [
        'Suggest off-peak usage benefits',
        'Provide time-based discount plans',
        'Offer peak hour usage optimization tips'
    ],
    'avg_monthly_usage_gb': [
        'Recommend right-sized data plans',
        'Provide data usage alerts',
        'Suggest family sharing plans if applicable'
    ],
    'service_outages_3months': [
        'Provide network improvement timeline',
        'Offer service credits for outages',
        'Install network signal boosters'
    ],
    'network_stability_score': [
        'Schedule technical support visit',
        'Upgrade network equipment',
        'Provide alternative connection options'
    ],
    'call_drops_per_month': [
        'Technical review of coverage area',
        'Offer WiFi calling features',
        'Provide network extender devices'
    ],
    'cs_tickets_3months': [
        'Assign dedicated support representative',
        'Provide priority support channel access',
        'Schedule regular check-in calls'
    ],
    'unresolved_tickets': [
        'Escalate to senior support team',
        'Schedule urgent resolution meeting',
        'Provide temporary service credits'
    ],
    'avg_resolution_time_hours': [
        'Fast-track support requests',
        'Implement automated issue tracking',
        'Provide alternative service options during resolution'
    ],
    'customer_engagement_score': [
        'Personalized feature tutorials',
        'Invite to customer feedback programs',
        'Offer exclusive service previews'
    ],
    'app_usage_frequency_change': [
        'Provide app usage incentives',
        'Simplify app interface if needed',
        'Offer digital service rewards'
    ],
    'feature_usage_breadth': [
        'Conduct feature awareness sessions',
        'Provide feature discovery rewards',
        'Create personalized feature guides'
    ],
    'service_downgrades_6months': [
        'Review service needs and preferences',
        'Offer trial of premium features',
        'Create custom service package'
    ],
    'service_cancellation_attempts': [
        'Schedule retention specialist call',
        'Offer competitive service comparison',
        'Provide loyalty incentives'
    ],
    'nps_score': [
        'Schedule satisfaction improvement call',
        'Provide premium service trial',
        'Offer personalized service package'
    ],
    'last_survey_rating': [
        'Follow-up on specific concerns',
        'Provide immediate service improvements',
        'Offer make-good compensation'
    ],
    'years_as_customer': [
        'Provide loyalty program benefits',
        'Offer long-term customer rewards',
        'Create personalized appreciation package'
    ],
    'historical_satisfaction_avg': [
        'Schedule service review meeting',
        'Create satisfaction improvement plan',
        'Offer premium service upgrade'
    ]
}

def generate_strategies_for_feature(feature):
    # Get predefined strategies from the mapping
    predefined_strategies = feature_to_strategy_mapping.get(feature, [])
    if not predefined_strategies:
        return ["No strategies found for the given feature."]
    
    try:
        # Format prompt for Groq
        prompt = f"Customer dissatisfaction is arising from the feature: {feature}. " \
                 "Please provide additional strategies along with the following predefined strategies to improve customer satisfaction:\n" + \
                 "\n".join(predefined_strategies) + \
                 "\nOutput only the strategies."

        messages = [
            {
                "role": "system",
                "content": "You are a helpful server agent focused on resolving customer dissatisfaction."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192"
        )
        
        # Get response and combine with predefined strategies
        ai_strategies = chat_completion.choices[0].message.content.split('\n')
        all_strategies = predefined_strategies + [s for s in ai_strategies if s not in predefined_strategies]
        
        return all_strategies
    
    except Exception as e:
        print(f"Error generating strategies: {str(e)}")
        return predefined_strategies  # Fallback to predefined strategies if API fails

def load_model_and_data():
    global model, X_test
    # Load your dataset
    df = pd.read_csv('new_synthesized_dataset.csv')
    
    # Prepare data
    X = df.drop(columns=['dissatisfaction', 'nps_score', 'last_survey_rating'])
    y = df['dissatisfaction'].to_numpy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgboost.XGBClassifier()
    model.fit(X_train, y_train)
    
    return X_train

def get_shap_analysis(customer_data):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(customer_data)
    print("SHAP Values:", shap_values[0].values)
    positive_contributors = []
    positive_values = []

    for i in range(len(shap_values[0].values)):
        if shap_values[0].values[i] > 0:
            positive_contributors.append(X_train.columns[i])
            positive_values.append(shap_values[0].values[i])
        
    # Get the highest influencing feature
    feature_importance = pd.DataFrame({
        'Feature': positive_contributors,
        'SHAP Value': positive_values
    })
    top_feature = feature_importance.nlargest(1, 'SHAP Value')['Feature'].iloc[0]
    
    return top_feature

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze_random_customers')
def analyze_customers():
    # Randomly select three customers
    random_indices = random.sample(range(len(X_test)), 3)
    customers = []
    
    for idx in random_indices:
        customer_data = X_test.iloc[[idx]]
        customer_id = f"C{1001 + idx}"  # Generate customer ID dynamically
        customer_name = f"Customer {idx + 1}"  # Placeholder names
        
        # Make prediction
        prediction = model.predict(customer_data)
        
        if prediction == 1:  # Dissatisfied
            top_feature = get_shap_analysis(customer_data)
            strategies = generate_strategies_for_feature(top_feature)[:3]  # Get only 3 strategies
            
            customers.append({
                'id': customer_id,
                'name': customer_name,
                'prediction': 'dissatisfied',
                'top_feature': top_feature,
                'strategies': strategies
            })
        else:
            customers.append({
                'id': customer_id,
                'name': customer_name,
                'prediction': 'satisfied',
                'strategies': []
            })
    
    return jsonify({'customers': customers})

from flask import jsonify, request
import shap

import shap
import matplotlib.pyplot as plt
import json

@app.route('/api/shap_force_plot/<customer_id>', methods=['GET'])
def shap_force_plot(customer_id):
    try:
        # Convert customer_id from C1234 format to int
        customer_id = int(customer_id.replace('C', ''))
        
        # Load SHAP explainer and precomputed SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test.iloc[[customer_id]])
        
        # Generate SHAP force plot
        plot_html = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[[customer_id]], show=False)
        
        # Convert the HTML to string and return as JSON
        return jsonify({'force_plot_html': str(plot_html)})
        
    except Exception as e:
        print(f"Error generating SHAP force plot: {str(e)}")
        return jsonify({'error': 'Failed to generate SHAP force plot'}), 500



if __name__ == '__main__':
    X_train = load_model_and_data()
    app.run(debug=True)

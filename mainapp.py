from flask import Flask, render_template, request, jsonify, send_file
import optimization_script
import pandas as pd
import os
import matplotlib.pyplot as plt



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    selected_stocks = data['stocks']
    risk_free_rate = float(data['risk_free_rate'])
    
    actual_columns = [
        'GUINESS Returns', 'UBA Returns', 'OANDO Returns', 
        'FBNH Returns', 'ETI Returns', 'CONOIL Returns', 'WAPCO Returns'
    ]
    
    if not all(stock in actual_columns for stock in selected_stocks):
        return jsonify({'error': 'Invalid stock selection'}), 400
    
    result = optimization_script.optimize_portfolio(selected_stocks, risk_free_rate)
    
    weights_df = pd.DataFrame.from_dict(result['weights'], orient='index', columns=['Weight'])
    weights_df.to_csv('optimized_weights.csv')

    response_data = {
        'return': f"{result['return']:.2%}",
        'volatility': f"{result['volatility']:.2%}",
        'sharpe_ratio': f"{result['sharpe_ratio']:.2f}",
        'weights': {k: f"{v:.2f}%" for k, v in result['weights'].items()},
        'timestamp': os.path.getmtime('efficient_frontier.png')
    }

    return jsonify(response_data)

@app.route('/download_csv')
def download_csv():
    return send_file('optimized_weights.csv', as_attachment=True)

@app.route('/efficient_frontier')
def efficient_frontier():
    return send_file('efficient_frontier.png', mimetype='image/png')

@app.route('/weights_pie_chart')
def weights_pie_chart():
    return send_file('weights_pie_chart.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

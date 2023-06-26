from flask import Flask
import functions as func

app = Flask(__name__)

@app.route('/recommendation/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    # Call the recommend_user_item function and return the recommendations
    items = func.recommend_user_item(int(user_id))
    return {'items': items}

@app.route('/item_simmilarity/<item_id>', methods=['GET'])
def get_similar_items(item_id):
    # Call the recommend_user_item function and return the recommendations
    items = func.get_similar_items(item_id)
    return {'items': items}


if __name__ == '__main__':
    app.run()
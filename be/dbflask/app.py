from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  
from os import environ
from decimal import Decimal
#from lg import main
#from lg import RAGsystem


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# AIChat message receiver

""" @app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    # Extract user input and session ID
    user_message = data.get("message")
    session_id = data.get("session_id")

    print(session_id)
    if not user_message or not session_id:
        return jsonify({"error": "Missing session_id or message"}), 400


    # Generate a response using RAGsystem
    response = RAGsystem(session_id, user_message)

    # Return the AI response
    return jsonify({"response": response["answer"]}) """


#DB setup

app.config['SQLALCHEMY_DATABASE_URI'] = environ.get('DATABASE_URL')
db = SQLAlchemy(app)

# Database Models
class Account(db.Model):
    __tablename__ = 'accounts'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())

class Client(db.Model):
    __tablename__ = 'clients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    phone = db.Column(db.String(20))
    address = db.Column(db.Text)
    account_id = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())

class Order(db.Model):
    __tablename__ = 'orders'
    id = db.Column(db.Integer, primary_key=True)
    order_date = db.Column(db.Date, default=db.func.current_date())
    status = db.Column(db.String(50), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('clients.id'), nullable=False)
    account_id = db.Column(db.Integer, db.ForeignKey('accounts.id'), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), server_onupdate=db.func.now())

class OrderItem(db.Model):
    __tablename__ = 'order_items'
    id = db.Column(db.Integer, primary_key=True)
    #order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    order_position = db.Column(db.Integer, nullable=False)
    product_id = db.Column(db.String(150), nullable=False)
    product_name = db.Column(db.String(150), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Numeric(15, 2), nullable=False)

# Initialize the Database Inside the App Context
with app.app_context():
    db.create_all()

  
# Initialize the Database Inside the App Context
with app.app_context():
    db.create_all()

# create a test route
@app.route('/test', methods=['GET'])
def test():
  return jsonify({'message': 'The server is running'})

# User Management API Endpoints
@app.route('/api/flask/accounts', methods=['POST'])
def create_account():
    try:
        data = request.get_json()
        new_account = Account(
            name=data['name'],
            email=data['email'],
            password_hash=data['password_hash']
        )
        db.session.add(new_account)
        db.session.commit()

        return jsonify({
            'id': new_account.id,
            'name': new_account.name,
            'email': new_account.email
        }), 201

    except Exception as e:
        return make_response(jsonify({'message': 'Error creating account', 'error': str(e)}), 500)

@app.route('/api/flask/clients', methods=['POST'])
def create_client():
    try:
        data = request.get_json()
        new_client = Client(
            name=data['name'],
            email=data['email'],
            phone=data['phone'],
            address=data['address'],
            account_id=data['account_id']
        )
        db.session.add(new_client)
        db.session.commit()

        return jsonify({
            'id': new_client.id,
            'name': new_client.name,
            'email': new_client.email
        }), 201

    except Exception as e:
        return make_response(jsonify({'message': 'Error creating client', 'error': str(e)}), 500)

@app.route('/api/flask/orders', methods=['POST'])
def create_order():
    try:
        data = request.get_json()
        new_order = Order(
            order_date=data['order_date'],
            status=data['status'],
            client_id=data['client_id'],
            account_id=data['account_id']
        )
        db.session.add(new_order)
        db.session.commit()

        return jsonify({
            'id': new_order.id,
            'status': new_order.status,
            'order_date': str(new_order.order_date)
        }), 201

    except Exception as e:
        return make_response(jsonify({'message': 'Error creating order', 'error': str(e)}), 500)

@app.route('/api/flask/order_items', methods=['POST'])
def create_order_item():
    try:
        data = request.get_json()
        new_order_item = OrderItem(
            #order_id=data["order_id"], 
            order_position=data["order_position"],
            product_id= data["product_id"],
            product_name= data["product_name"],
            quantity= int(data["quantity"]),
            value= Decimal(f'{data["value"]:.2f}')
        )
        db.session.add(new_order_item)
        db.session.commit()

        return jsonify({
            'product_id': new_order_item.product_id,
            'quantity': new_order_item.quantity,
            'value': new_order_item.value
        }), 201

    except Exception as e:
        return make_response(jsonify({'message': 'Error creating order item', 'error': str(e)}), 500)



#----------------------------------TO BE MODIFIED-------------------------------------

# create a user
@app.route('/api/flask/users', methods=['POST'])
def create_user():
  try:
    data = request.get_json()
    new_user = User(name=data['name'], email=data['email'])
    db.session.add(new_user)
    db.session.commit()  

    return jsonify({
        'id': new_user.id,
        'name': new_user.name,
        'email': new_user.email
    }), 201  

  except Exception as e:
    return make_response(jsonify({'message': 'error creating user', 'error': str(e)}), 500)
  
# get all users
@app.route('/api/flask/users', methods=['GET'])
def get_users():
  try:
    users = User.query.all()
    users_data = [{'id': user.id, 'name': user.name, 'email': user.email} for user in users]
    return jsonify(users_data), 200
  except Exception as e:
    return make_response(jsonify({'message': 'error getting users', 'error': str(e)}), 500)
  
# get a user by id
@app.route('/api/flask/users/<id>', methods=['GET'])
def get_user(id):
  try:
    user = User.query.filter_by(id=id).first() # get the first user with the id
    if user:
      return make_response(jsonify({'user': user.json()}), 200)
    return make_response(jsonify({'message': 'user not found'}), 404) 
  except Exception as e:
    return make_response(jsonify({'message': 'error getting user', 'error': str(e)}), 500)
  
# update a user by id
@app.route('/api/flask/users/<id>', methods=['PUT'])
def update_user(id):
  try:
    user = User.query.filter_by(id=id).first()
    if user:
      data = request.get_json()
      user.name = data['name']
      user.email = data['email']
      db.session.commit()
      return make_response(jsonify({'message': 'user updated'}), 200)
    return make_response(jsonify({'message': 'user not found'}), 404)  
  except Exception as e:
      return make_response(jsonify({'message': 'error updating user', 'error': str(e)}), 500)

# delete a user by id
@app.route('/api/flask/users/<id>', methods=['DELETE'])
def delete_user(id):
  try:
    user = User.query.filter_by(id=id).first()
    if user:
      db.session.delete(user)
      db.session.commit()
      return make_response(jsonify({'message': 'user deleted'}), 200)
    return make_response(jsonify({'message': 'user not found'}), 404) 
  except Exception as e:
    return make_response(jsonify({'message': 'error deleting user', 'error': str(e)}), 500)   


if __name__ == "__main__":
    app.run(port=4000)

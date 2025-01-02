from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  
from os import environ

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database Setup
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
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    product_id = db.Column(db.String(150), nullable=False)
    product_name = db.Column(db.String(150), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)

# Initialize the Database Inside the App Context
with app.app_context():
    db.create_all()

# Test Route
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

            order_id=data["order"] 
            order_item=data["order_item"] 
            product_id= data["product_id"]
            product_name= data["product_name"]
            quantity= data["quantity"]
            price= ["price"]

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



if __name__ == "__main__":
    app.run(port=4000)

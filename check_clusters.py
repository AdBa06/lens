from database import SessionLocal
from models import Cluster

db = SessionLocal()
clusters = db.query(Cluster).all()
print('Current cluster breakdown:')
for c in clusters:
    print(f'Cluster {c.id}: {c.size} events, algorithm: {c.cluster_algorithm}')
db.close() 
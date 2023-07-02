import pypyodbc # pip install pypyodbc
import pandas as pd # pip install pandas

SERVER_NAME = 'localhost'
DATABASE_NAME = 'AutomationMachine'

conn = pypyodbc.connect("""
    Driver={{SQL Server Native Client 11.0}};
    Server={0};
    Database={1};
    Trusted_Connection=yes;""".format(SERVER_NAME, DATABASE_NAME)
)


imlec=conn.cursor()


def getProductDemo():
    sql_query = """
    Select * from ProductDemos WITH (NoLock)
    """
    return pd.read_sql(sql_query, conn)

def getRealProducts():
    sql_query = """
    Select * from Products WITH (NoLock)
    """
    return pd.read_sql(sql_query, conn)

def addProductDemo(heat,materialSpeed,fanSpeed,quality):
    command = 'INSERT INTO ProductDemos VALUES(?,?,?,?)'
    setData = (heat,materialSpeed,fanSpeed,quality)
    result = imlec.execute(command, setData)
    conn.commit()








from threadpoolctl import threadpool_limits
from full_embedding_ridge import main
if __name__=='__main__':
    with threadpool_limits(limits=2):main()

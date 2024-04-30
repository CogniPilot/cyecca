double Chirp(double w1, double w2, double A, double M, double time)
{
    double res;
    res=A*(w1-w2)*M*time;
    return res;
}

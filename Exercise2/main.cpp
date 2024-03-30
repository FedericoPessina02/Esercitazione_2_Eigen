#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

double relerr(const Eigen::Vector2d& x0, const Eigen::Vector2d& x1) {
    double err = (x0-x1).norm()/x0.norm();
    return err;
}

Eigen::Vector2d luSolution(const Eigen::Matrix2d& A, const Eigen::Vector2d& b) {
    // non è stato utilizzato il metodo del pivoting parziale perché le matrici 2 e 3 hanno determinante vicino allo 0
    // Eigen::PartialPivLU<Eigen::Matrix2d> lu(A);
    Eigen::FullPivLU<Eigen::Matrix2d> lu(A);
    Eigen::Vector2d x =  lu.solve(b);
    return x;
}

Eigen::Vector2d qrSolution(const Eigen::Matrix2d& A, const Eigen::Vector2d& b) {
    Eigen::ColPivHouseholderQR<Eigen::Matrix2d> qr(A);
    Eigen::Vector2d x = qr.solve(b);
    return x;
}

void testMethods(const Eigen::Matrix2d& A, const Eigen::Vector2d& b, std::string testLabel) {
    Eigen::Vector2d luRes = luSolution(A, b);
    Eigen::Vector2d qrRes = qrSolution(A, b);
    Eigen::Vector2d realval(-1.0e+0, -1.0e+00);
    double luErr = relerr(luRes, realval);
    double qrErr = relerr(qrRes, realval);
    std::cout << testLabel << " - risultati" << std::endl;
    std::cout << std::scientific << std::setprecision(16) << "A: " << A << std::endl;
    std::cout << std::scientific << std::setprecision(16) << "b: " << b << std::endl;
    std::cout << std::scientific << std::setprecision(16) << "    Fattorizzazione LU errore: " << luErr << std::endl;
    std::cout << std::scientific << std::setprecision(16) << "    Fattorizzazione QR errore: " << qrErr << std::endl;
    std::cout << std::endl;
}

int main()
{
    Eigen::Matrix2d A0;
    A0 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    Eigen::Vector2d b0(-5.169911863249772e-01, 1.672384680188350e-01);
    testMethods(A0, b0, "Sistema 1");

    Eigen::Matrix2d A1;
    A1 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    Eigen::Vector2d b1(-6.394645785530173e-04, 4.259549612877223e-04);
    testMethods(A1, b1, "Sistema 2");

    Eigen::Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    Eigen::Vector2d b2(-6.400391328043042e-10, 4.266924591433963e-10);
    testMethods(A2, b2, "Sistema 3");
    return 0;
}
